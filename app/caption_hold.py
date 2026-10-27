"""Hold a segment until the scene caption its prompt needs exists (console#562).

WHAT WAS WRONG

    A pose fills <SCENE> from its start image's saved caption (see _resolve_scene in
    app/routes/segments.py). Before this, <SCENE> on a cache miss was captioned AGAIN,
    synchronously, inside the submit. The submit blocked for the length of a caption, and the
    words it baked in were not the ones the image modal was producing at that moment -- two
    captions of one image, two different descriptions, and the render used whichever the API
    happened to make.

WHAT THIS DOES

    A segment whose prompt needs a scene that is not saved yet is created AWAITING_CAPTION
    instead of PENDING, and the submit returns at once. The claim endpoint only ever hands
    out PENDING, so a held segment is not claimable by construction -- no daemon change, no
    race with the claim.

    One background waiter per IMAGE (not per segment) then gets the words:

      1. JOIN. If a caption of that image is already queued or running -- the modal's, the
         New Job dialog's, another job's -- it waits for that one to finish instead of
         starting its own. A second caption would overwrite the first with different words,
         and the whole point is to render the words that were shown.
      2. Otherwise it asks for one. Since console#564 both are the same call: a caption
         TICKET (app/caption_tickets.py), single-flight per image, which takes a turn in the
         ordinary caption queue -- the same line the modal stands in. A describe clicked
         while the job's ticket is queued joins that ticket rather than queueing a second.
      3. RELEASE. When the scene a held segment needs is saved on the image's ImageMeta row,
         the placeholder is filled from THAT row -- exactly the words the lightbox shows --
         and the segment goes to PENDING.

    Failure is loud. A caption error, an unreadable image, or the captioner refusing for
    longer than caption_hold_timeout_s moves the segment to CAPTION_FAILED with the reason in
    error_message, and it stays there until a person picks Retry caption or Render without
    (routes in app/routes/segments.py). Nothing here ever drops the scene on its own.

    State lives in the database, so a restart loses nothing but the in-memory waiters, and
    caption_hold_monitor re-creates those: once at startup and every caption_hold_sweep_s.
    The sweep is also what releases a segment whose words were saved by something this
    module never saw.

WHAT IT DOES NOT CHANGE

    * Continuations whose start image is not known yet (start_image NULL, index > 0) keep the
      deferral: the frame does not exist until the previous segment renders, so there is
      nothing to wait on. The claim resolves them exactly as before.

<MOTION> IS GONE (wanly-api#481)

    There used to be a second, motion half with its own placeholder, gated here like the
    scene. It was removed. A prompt that still carries <MOTION> (an old pose, a segment held
    before the removal) never waits for it: the placeholder is stripped at submit, at release
    and at the claim (segments._strip_legacy_motion), never held and never sent as text.

THE PLACEHOLDER HAS TO ARRIVE

    Everything here keys on a literal <SCENE> in the submitted prompt. The console used to
    delete an unfilled one before submitting, so this never fired and a recipe queued
    mid-caption rendered with an empty prompt (console#577). It now sends the bare
    placeholder; the API refuses a blank one at submit and at the claim either way.

NEVER BLOCKS THE LOOP, NEVER HOLDS A CONNECTION ACROSS A CAPTION

    The S3 read goes through asyncio.to_thread. Every database touch is its own short
    session, opened and closed around a few statements; none is open while the captioner is
    working or while waiting in the queue, so a long caption cannot drain the pool
    (console#559 is that failure).
"""
from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy import and_, or_, select

from app import caption_tickets as tickets
from app import s3  # noqa: F401 - tests patch s3.download_bytes through this name
from app.caption_queue import queue as caption_queue
from app.config import settings
from app.database import async_session
from app.enums import SegmentStatus
from app.models import ImageMeta, Job, Segment

logger = logging.getLogger(__name__)

SCENE = "scene"

#: How often a waiter looks at the queue while joining a describe-kind turn that is not a
#: caption ticket. Tickets are awaited directly; this is only the fallback.
JOIN_POLL_S = 1.0
#: How long to wait before asking again when the captioner refused because the box beside
#: it is rendering. A render is minutes; asking every second would only fill the log.
BUSY_RETRY_S = 30.0

#: The waiter for each image, by URI. The single-flight half that lives in this process: the
#: sweep, a retry and a second job on the same image all get the waiter that is already
#: there instead of starting another.
_waiters: dict[str, asyncio.Task] = {}
#: What each waiter is doing right now, for the console. Never stored: it is only true while
#: the waiter is alive.
_notes: dict[str, str] = {}


def _seg():
    # Deferred: app.routes.segments imports this module, and the placeholder rules have one
    # home, there.
    from app.routes import segments
    return segments


# ---------------------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------------------

def needed_halves(prompt: str | None) -> set[str]:
    """The caption halves this prompt must wait for: the scene, whenever it carries <SCENE>.

    A legacy <MOTION> is never waited for (wanly-api#481); it is stripped instead.
    """
    return {SCENE} if _seg().SCENE_PLACEHOLDER in (prompt or "") else set()


def saved_halves(meta: ImageMeta | None) -> dict[str, str]:
    """The words saved on an image's row, by half. A blank column is not a saved half."""
    out: dict[str, str] = {}
    if meta is not None and (meta.scene_description or "").strip():
        out[SCENE] = meta.scene_description
    return out


def fill_saved(prompt: str, meta: ImageMeta | None) -> str:
    """Fill <SCENE> if it has saved words, leaving it literal otherwise; strip any legacy
    <MOTION> (wanly-api#481).

    str.replace does not rescan what it inserted, so caption words that happen to contain
    "<SCENE>" stay words. The text is the row's, verbatim: what the lightbox shows is what
    renders.
    """
    seg = _seg()
    prompt = seg._strip_legacy_motion(prompt)
    scene = saved_halves(meta).get(SCENE)
    return prompt.replace(seg.SCENE_PLACEHOLDER, scene) if scene is not None else prompt


def hold_image(start_image: str | None, index: int, job_starting_image: str | None) -> str | None:
    """The frame a segment's captions come from, when it is known before the claim.

    Its own start image if it has one; segment 0 otherwise starts from the job's. A later
    segment with none continues from a frame that does not exist yet -- None, and no hold.
    """
    if start_image:
        return start_image
    if index == 0:
        return job_starting_image
    return None


def _held_at(path: str):
    """SQL: segments whose hold image is `path` -- hold_image, as a WHERE clause."""
    return or_(
        Segment.start_image == path,
        and_(Segment.start_image.is_(None), Segment.index == 0, Job.starting_image == path),
    )


# ---------------------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------------------

async def gate(db, prompt: str, image_uri: str | None) -> tuple[str, bool] | None:
    """Decide at submit whether a segment is held. Returns (prompt, held), or None.

    None means "not a case this module handles; do what was done before": no image known
    yet (a continuation -- deferred to the claim), a frame outside our buckets (its words
    could never be saved, so there is nothing to wait FOR), or a prompt with no half to wait
    on.

    (prompt, False): every needed half is saved, and the prompt comes back filled from the
    row. (prompt, True): hold it; the caller creates the segment AWAITING_CAPTION and calls
    ensure(image_uri) once it has committed.

    A caption of this image IN FLIGHT holds the segment even if words are already saved. That
    is a re-roll in progress, and the words about to land are the ones on the person's
    screen; the saved ones are about to be replaced.
    """
    if not image_uri or not _seg()._is_describable(image_uri):
        return None
    needs = needed_halves(prompt)
    if not needs:
        return None
    meta = await db.get(ImageMeta, image_uri)
    missing = needs - saved_halves(meta).keys()
    running = caption_in_flight(image_uri, needs)
    if not missing and not running:
        return fill_saved(prompt, meta), False
    logger.info("Caption hold: holding a segment on %s until its %s %s saved%s",
                image_uri, " and ".join(sorted(missing or needs)),
                "are" if len(missing or needs) > 1 else "is",
                " (a caption of it is running)" if running else "")
    return prompt, True


def caption_in_flight(path: str, halves=None) -> bool:
    """Is a caption that will SAVE words on this image queued or running?

    A caption ticket (app/caption_tickets.py) of one of `halves` (default: either), or -- for
    anything that takes a queue turn without one -- a describe-kind turn. Dataset captions and
    Settings tries take turns too, but write nothing on the image's row, so they are not worth
    """
    halves = tuple(halves) if halves else tickets.HALVES
    return any(tickets.active(path, h) is not None for h in halves) or caption_queue.in_flight(
        path, kinds=WRITES_WORDS)


#: Queue kinds whose caption lands on the image's ImageMeta row.
WRITES_WORDS = frozenset({"describe", "hold"})


# ---------------------------------------------------------------------------------------
# Release and failure
# ---------------------------------------------------------------------------------------

async def settle(path: str, finishing=None) -> set[str]:
    """Release every held segment on `path` whose halves are all saved.

    Returns the halves still outstanding across the ones left waiting; empty means nothing
    on this image is held any more (all released, or none were). One short session, rows
    locked so a Retry or Render-without on the same segment cannot interleave.

    A segment whose scene a person is re-rolling right now is NOT released, though its words
    are saved (gate()'s rule): the re-rolled words are
    the ones on the person's screen. Its hold waits for that re-roll. `finishing` is the
    ticket calling this as it saves -- it is still in flight, but it is not a re-roll to wait
    for any more.
    """
    async with async_session() as db:
        rows = (await db.execute(
            select(Segment)
            .join(Job, Segment.job_id == Job.id)
            .where(Segment.status == SegmentStatus.AWAITING_CAPTION,
                   Segment.discarded.is_(False), _held_at(path))
            .with_for_update(of=Segment)
        )).scalars().all()
        if not rows:
            return set()
        meta = await db.get(ImageMeta, path)
        have = saved_halves(meta).keys()
        rerolling = {h for h in tickets.HALVES
                     if tickets.active(path, h) is not finishing and _needs_now(path, h)}
        outstanding: set[str] = set()
        for s in rows:
            needs = needed_halves(s.prompt)
            missing = needs - have
            if missing:
                outstanding |= missing
                continue
            if needs & rerolling:
                continue  # its hold is waiting for the re-roll
            s.prompt = fill_saved(s.prompt, meta)
            s.status = SegmentStatus.PENDING
            s.error_message = None
            s.progress_log = None
            logger.info("Caption hold: released segment %s (job %s, index %d) with the saved "
                        "caption of %s", s.id, s.job_id, s.index, path)
        await db.commit()
        return outstanding


async def fail(path: str, reason: str) -> int:
    """Move every segment held on `path` to CAPTION_FAILED, saying why. Returns how many."""
    async with async_session() as db:
        rows = (await db.execute(
            select(Segment)
            .join(Job, Segment.job_id == Job.id)
            .where(Segment.status == SegmentStatus.AWAITING_CAPTION,
                   Segment.discarded.is_(False), _held_at(path))
            .with_for_update(of=Segment)
        )).scalars().all()
        for s in rows:
            s.status = SegmentStatus.CAPTION_FAILED
            s.error_message = f"Caption failed: {reason}"
            s.progress_log = None
        await db.commit()
    if rows:
        logger.warning("Caption hold: %d segment(s) on %s -> caption_failed: %s",
                       len(rows), path, reason)
    return len(rows)


# ---------------------------------------------------------------------------------------
# The waiter
# ---------------------------------------------------------------------------------------

def ensure(path: str) -> asyncio.Task:
    """Make sure something is getting `path` its caption. Idempotent; returns the waiter."""
    task = _waiters.get(path)
    if task is not None and not task.done():
        return task
    task = asyncio.create_task(_hold(path), name=f"caption-hold {path}")
    _waiters[path] = task

    def _forget(t: asyncio.Task, p: str = path) -> None:
        if _waiters.get(p) is t:
            del _waiters[p]
    task.add_done_callback(_forget)
    return task


def hold_place(path: str) -> dict:
    """Where the caption a held image is waiting on has got to, right now.

    {"queue_status": "queued" | "running" | "waiting" | None, "queue_position", "queue_depth",
    "note"}. "waiting" is a hold with nothing in the queue: between a refusal and asking
    again (the box beside the captioner is rendering), or about to ask. None: no waiter at
    all -- the sweep gives it one within caption_hold_sweep_s.
    """
    t = tickets.active(path)
    depth = caption_queue.depth()
    note = _notes.get(path)
    if t is not None:
        v = tickets.view(t)
        return {"queue_status": v["status"], "queue_position": v["position"],
                "queue_depth": v["depth"], "note": note, "lane": v.get("lane")}
    st = caption_queue.status(path)
    if st["status"] is not None and caption_queue.in_flight(path, kinds=WRITES_WORDS):
        return {"queue_status": st["status"], "queue_position": st["position"],
                "queue_depth": depth, "note": note, "lane": "scene"}
    task = _waiters.get(path)
    alive = task is not None and not task.done()
    return {"queue_status": "waiting" if alive else None, "queue_position": None,
            "queue_depth": depth, "note": note, "lane": None}


#: Images whose hold is waiting because no box is in the mode the scene's fallback captioner
#: needs (wanly-api#392: the scene service down and no box in "motion" mode), with the reason.
_mode_waits: dict[str, str] = {}


def mode_waiting() -> dict[str, str]:
    """{image path: reason} for every hold waiting on a mode no box is in."""
    return dict(_mode_waits)


def wait_note(path: str) -> str | None:
    """What the hold on this image is waiting for, in words, with its place in the queue."""
    place = hold_place(path)
    note = place["note"]
    if place["queue_status"] == "running":
        where = "captioning now"
    elif place["queue_status"] == "queued" and place["queue_position"]:
        where = f"#{place['queue_position']} of {place['queue_depth']} in the caption queue"
    else:
        where = None
    parts = [p for p in (note, where) if p]
    return "; ".join(parts) if parts else None


def missing_halves(prompt: str | None, meta: ImageMeta | None) -> list[str]:
    """What a held prompt is still waiting for: ["scene"] or []."""
    missing = needed_halves(prompt) - saved_halves(meta).keys()
    return [h for h in (SCENE,) if h in missing]


async def held_jobs(path: str) -> tuple[set[str], dict[str, list[dict]]]:
    """What the segments held on `path` use, and the jobs by the half each still needs:
    ({half}, {half: [{"job_id", "name"}]}).

    The jobs are what a caption ticket the hold asks for carries as `requested_by`, so the
    image can say "requested by job ..." (console#590) -- a caption nobody clicked for.
    """
    async with async_session() as db:
        rows = (await db.execute(
            select(Segment.prompt, Job.id, Job.name)
            .join(Job, Segment.job_id == Job.id)
            .where(Segment.status == SegmentStatus.AWAITING_CAPTION,
                   Segment.discarded.is_(False), _held_at(path))
            .order_by(Segment.created_at)
        )).all()
        if not rows:
            return set(), {}
        meta = await db.get(ImageMeta, path)
    have = saved_halves(meta).keys()
    used: set[str] = set()
    out: dict[str, list[dict]] = {}
    for prompt, job_id, name in rows:
        used |= needed_halves(prompt)
        for half in needed_halves(prompt) - have:
            jobs = out.setdefault(half, [])
            if all(j["job_id"] != str(job_id) for j in jobs):
                jobs.append({"job_id": str(job_id), "name": name})
    return used, out


async def _hold(path: str) -> None:
    """Get `path` its words and release what is held on it.

    Everything goes through caption tickets (app/caption_tickets.py), the same single-flight
    table describe uses, so "join the caption already running" and "make one" are the same
    call: request() hands back the image's scene ticket if there is one.

    TIME LIMIT. caption_hold_timeout_s bounds how long the captioner may keep REFUSING (the
    box beside it rendering), counted from the first refusal in a row. It does not bound time
    spent in line: a caption that is queued is a caption that is coming, and a held job that
    failed because twenty images were ahead of it would be failing for nothing (console#562's
    first version did exactly that when joining a describe deep in the queue).
    """
    busy_since: float | None = None
    try:
        while True:
            _mode_waits.pop(path, None)
            if tickets.active(path) is None and caption_queue.in_flight(
                    path, kinds=WRITES_WORDS):
                # A describe-kind turn that is not a ticket. Nothing in the API takes one any
                # more, but joining is cheap and a second caption is what this exists to stop.
                _notes[path] = "waiting for the caption already running for this image"
                while caption_queue.in_flight(path, kinds=WRITES_WORDS):
                    await asyncio.sleep(JOIN_POLL_S)
                continue

            used, jobs = await held_jobs(path)
            if not used:
                # Nothing held here needs a caption any more -- or only a legacy <MOTION> did
                # (wanly-api#481), which settle() strips on release.
                await settle(path)
                return
            # A person's re-roll of a half these segments use is waited for, never released
            # ahead of (gate()'s rule): its words are about to be on the person's screen.
            rerolls = [tickets.active(path, h) for h in (SCENE,)
                       if h in used and _needs_now(path, h)]
            if rerolls:
                _notes[path] = "waiting for the caption already running for this image"
                for t in rerolls:
                    t.add_requesters(jobs.get(t.half))
                await asyncio.gather(*(t.wait() for t in rerolls))
                continue  # whatever happened to it, look again

            outstanding = await settle(path)
            if not outstanding:
                return
            waits: list[tuple[str, tickets.Ticket, bool]] = []
            for half in (SCENE,):
                if half in outstanding:
                    t, joined = tickets.request(
                        path, half, origin="hold", precheck=_precheck(path, half),
                        requested_by=jobs.get(half))
                    waits.append((half, t, not joined))
            _notes[path] = "waiting for the " + " and ".join(h for h, _, _ in waits)

            await asyncio.gather(*(t.wait() for _, t, _ in waits))

            refused: str | None = None
            #: Every refusal was "no GPU in <mode> mode" (wanly-api#392): no time limit.
            mode_only = True
            for half, t, own in waits:
                if t.status != tickets.FAILED or t.withdrawn:
                    continue
                if t.busy:
                    # The box beside the captioner is rendering. Not a failure -- the modal
                    # is refused the same way -- so wait it out, up to the limit.
                    refused = refused or t.error
                    mode_only = mode_only and t.mode_wait
                elif own:
                    await fail(path, f"the {half} caption failed: {t.error or 'no reason given'}")
                    return
                else:
                    # Somebody else's caption of this image failed. Make our own.
                    logger.info("Caption hold: the %s caption %s was waiting on failed (%s); "
                                "asking for its own", half, path, t.error)
            if refused is not None and mode_only:
                # No box is in the mode this caption needs. Not the captioner refusing -- there
                # is no captioner to refuse -- so the hold's time limit does not apply: the
                # person switches a box (the reason says how), and the next ask goes to it.
                busy_since = None
                # The bare reason: the console prefixes "Waiting: " to a hold's note.
                _notes[path] = refused
                _mode_waits[path] = refused
                logger.info("Caption hold: %s waits (%s); asking again in %ds", path, refused,
                            BUSY_RETRY_S)
                await asyncio.sleep(BUSY_RETRY_S)
                continue
            if refused is not None:
                busy_since = busy_since if busy_since is not None else time.monotonic()
                if not await _wait_out_refusal(path, refused, busy_since):
                    return
                continue
            busy_since = None
            if any(tickets.active(path, h) is not None for h, _, _ in waits):
                # Another caption of a half was queued behind that one -- a re-roll. Its
                # words are the ones about to be on the person's screen.
                continue
            left = await settle(path)
            if not left:
                return
            mine = [h for h, t, own in waits
                    if own and h in left and t.status == tickets.DONE and not t.skipped]
            if mine:
                await fail(path, f"the caption was made but its {' and '.join(mine)} half is "
                                 "still missing")
                return
            # Somebody else's caption landed without the half we need: ask for it ourselves.
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - whatever it was, a person must hear about it
        logger.exception("Caption hold on %s failed", path)
        await fail(path, f"{type(e).__name__}: {e}")
    finally:
        _notes.pop(path, None)
        _mode_waits.pop(path, None)


def _needs_now(path: str, half: str) -> bool:
    """Does the ticket in flight for this half count as a re-roll the hold must wait for?
    A person's (or one a describe joined); the hold's own does not."""
    t = tickets.active(path, half)
    return t is not None and (t.origin != "hold" or t.precheck is None)


async def _wait_out_refusal(path: str, why: str | None, busy_since: float) -> bool:
    """Sleep BUSY_RETRY_S before asking again; False (and the hold failed) past the limit."""
    if time.monotonic() - busy_since >= settings.caption_hold_timeout_s:
        await fail(path, f"{_timed_out()} ({why})")
        return False
    _notes[path] = f"the captioner is unavailable: {why}"
    logger.info("Caption hold: %s refused (%s); asking again in %ds", path, why, BUSY_RETRY_S)
    await asyncio.sleep(BUSY_RETRY_S)
    return True


def _precheck(path: str, half: str):
    """A hold ticket's precheck, at the front of the line: release what can be, and say
    whether this half is no longer needed. Whatever was ahead of it may have been this
    image."""
    async def check() -> bool:
        return half not in await settle(path)
    return check


def _timed_out() -> str:
    return (f"no caption after {settings.caption_hold_timeout_s // 60} minutes of the "
            f"captioner refusing")


# ---------------------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------------------

async def sweep() -> int:
    """Give every held image a waiter. Returns how many images are held.

    What survives a restart is the AWAITING_CAPTION rows; the waiters do not. This re-creates
    them, and because a waiter's first act is to settle, it also releases anything whose
    words were saved while nobody was watching. A held row whose image cannot be captioned
    at all is failed with the reason rather than left to wait forever.
    """
    async with async_session() as db:
        rows = (await db.execute(
            select(Segment.id, Segment.start_image, Segment.index, Job.starting_image)
            .join(Job, Segment.job_id == Job.id)
            .where(Segment.status == SegmentStatus.AWAITING_CAPTION,
                   Segment.discarded.is_(False))
        )).all()
    paths: set[str] = set()
    orphans = []
    for seg_id, start_image, index, job_start in rows:
        path = hold_image(start_image, index, job_start)
        if path and _seg()._is_describable(path):
            paths.add(path)
        else:
            orphans.append(seg_id)
    if orphans:
        async with async_session() as db:
            for seg_id in orphans:
                s = await db.get(Segment, seg_id)
                if s is not None and s.status == SegmentStatus.AWAITING_CAPTION:
                    s.status = SegmentStatus.CAPTION_FAILED
                    s.error_message = ("Caption failed: this segment has no start image the "
                                       "API can caption")
            await db.commit()
        logger.warning("Caption hold: %d held segment(s) have no captionable start image",
                       len(orphans))
    released = 0
    for path in paths:
        if _waits_on_a_reroll(path):
            ensure(path)
            continue
        # SETTLE, not just ensure (the console#562 follow-up). A waiter only looks at the row
        # when its own caption finishes or reaches the front of the line, and the line can be
        # hours long. Words saved some other way meanwhile -- the bulk-tag auto-describe, a
        # caption that finished for a different reason, a restart -- left the segment held
        # behind captions that had nothing to do with it. Seen live on 2026-10-02: a deploy's
        # startup sweep released two such segments at once, words long since saved.
        try:
            outstanding = await settle(path)
        except Exception:  # noqa: BLE001 - one image's bad row must not stop the sweep
            logger.exception("Caption hold: sweep could not settle %s", path)
            ensure(path)
            continue
        if not outstanding:
            released += 1
            for half in tickets.HALVES:
                t = tickets.active(path, half)
                if t is not None and t.origin == "hold" and t.status == tickets.QUEUED:
                    tickets.withdraw(t, "nothing on this image needs a caption any more")
            continue
        ensure(path)
    if released:
        logger.info("Caption hold: the sweep released everything held on %d image(s) whose "
                    "words were already saved", released)
    return len(paths)


def _waits_on_a_reroll(path: str) -> bool:
    """Is a caption of this image in flight that the hold must NOT release ahead of?

    A describe (a person's re-roll) -- its words, not the saved ones, are about to be on the
    screen, which is gate()'s rule. The hold's OWN ticket does not count, unless a describe
    has joined it (which drops its precheck): then it is a re-roll too.
    """
    if any(tickets.active(path, h) is not None for h in tickets.HALVES):
        return any(_needs_now(path, h) for h in tickets.HALVES)
    return caption_queue.in_flight(path, kinds=WRITES_WORDS)


async def caption_hold_monitor() -> None:
    """Startup sweep, then one every caption_hold_sweep_s. Lives for the app's lifetime."""
    first = True
    while True:
        try:
            n = await sweep()
            if first and n:
                logger.info("Caption hold: resumed waiting on %d image(s) after startup", n)
        except Exception:  # noqa: BLE001 - a bad sweep must not end the sweeping
            logger.exception("Caption hold sweep failed")
        first = False
        await asyncio.sleep(settings.caption_hold_sweep_s)
