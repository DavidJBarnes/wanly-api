"""Hold a segment until the caption its prompt needs exists (console#562).

WHAT WAS WRONG

    A pose fills <SCENE> and <MOTION> from its start image's saved captions (see
    _resolve_scene in app/routes/segments.py). Before this, the two halves failed in two
    different ways, and both were silent:

      * <SCENE> on a cache miss was captioned AGAIN, synchronously, inside the submit. The
        submit blocked for the length of a caption, and the words it baked in were not the
        ones the image modal was producing at that moment -- two captions of one image, two
        different descriptions, and the render used whichever the API happened to make.
      * <MOTION> is cache-only and, if nothing was saved by the time a worker claimed the
        segment, was DROPPED. Caption an image, click Use as Starting Image, submit: if the
        3090 claimed before the motion paragraph landed, the render ran without it and
        nothing said so.

WHAT THIS DOES

    A segment whose prompt needs a caption half that is not saved yet is created
    AWAITING_CAPTION instead of PENDING, and the submit returns at once. The claim endpoint
    only ever hands out PENDING, so a held segment is not claimable by construction -- no
    daemon change, no race with the claim.

    One background waiter per IMAGE (not per segment) then gets the words:

      1. JOIN. If a caption of that image is already queued or running -- the modal's, the
         New Job dialog's, another job's -- it waits for that one to finish instead of
         starting its own. A second caption would overwrite the first with different words,
         and the whole point is to render the words that were shown.
      2. Otherwise it takes a turn in the ordinary caption queue (app/caption_queue.py), the
         same line the modal stands in, and re-checks once it is at the front: whatever was
         ahead of it may have been this image.
      3. RELEASE. When every half a held segment needs is saved on the image's ImageMeta row,
         the placeholders are filled from THAT row -- exactly the words the lightbox shows --
         and the segment goes to PENDING.

    Failure is loud. A caption error, an unreadable image, or waiting past
    caption_hold_timeout_s moves the segment to CAPTION_FAILED with the reason in
    error_message, and it stays there until a person picks Retry caption or Render without
    (routes in app/routes/segments.py). Nothing here ever drops a half on its own.

    State lives in the database, so a restart loses nothing but the in-memory waiters, and
    caption_hold_monitor re-creates those: once at startup and every caption_hold_sweep_s.
    The sweep is also what releases a segment whose words were saved by something this
    module never saw.

WHAT IT DOES NOT CHANGE

    * Continuations whose start image is not known yet (start_image NULL, index > 0) keep the
      deferral: the frame does not exist until the previous segment renders, so there is
      nothing to wait on. The claim resolves them exactly as before.
    * <MOTION> gates whenever the prompt carries it and motion captioning can run
      (motion_captioning_can_run: MOTION_CAPTION_ENABLED, true by default). Only with the
      kill-switch off does <SCENE> gate alone -- then <MOTION> is filled if a paragraph
      happens to be saved and is otherwise deferred to the claim and dropped there, because
      waiting for a half that is switched off would be waiting forever. The claim refuses a
      prompt that the drop leaves empty or trigger-only (console#577), so that case is loud.

THE PLACEHOLDER HAS TO ARRIVE

    Everything here keys on a literal <SCENE>/<MOTION> in the submitted prompt. The console
    used to delete an unfilled one before submitting, so this never fired and a Motion recipe
    queued mid-caption rendered with an empty prompt (console#577). It now sends the bare
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
from datetime import datetime, timezone

from sqlalchemy import and_, or_, select

from app import s3
from app.caption_queue import queue as caption_queue
from app.config import settings
from app.database import async_session
from app.enums import SegmentStatus
from app.joycaption import CaptionError, CaptionerBusy, describe_motion
from app.models import ImageMeta, Job, Segment

logger = logging.getLogger(__name__)

SCENE = "scene"
MOTION = "motion"

#: How often a waiter looks at the queue while joining someone else's caption. The queue is
#: in this process, so a look costs nothing; the wait it is watching is ~25 s a call.
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

def motion_captioning_can_run() -> bool:
    """Can a caption of an image produce the motion paragraph at all?

    MOTION_CAPTION_ENABLED, the env kill-switch from #326 -- true by default, and set false
    only for a captioner whose model cannot answer the directional prompt. While it is on,
    every caption this module makes (run_caption_pair / describe_motion) produces the
    paragraph, so waiting for one is waiting for something that will come.
    """
    return bool(settings.motion_caption_enabled)


def needed_halves(prompt: str | None) -> set[str]:
    """The caption halves this prompt must wait for.

    <SCENE> whenever the prompt carries it. <MOTION> whenever the prompt carries it and
    motion captioning can run -- a Motion recipe's prompt is little more than its trigger
    phrase and <MOTION>, so rendering it without the paragraph renders nothing (console#577).
    With the kill-switch off no caption will ever produce the paragraph, so waiting for one
    would be waiting forever; the placeholder keeps its pre-#562 behaviour instead (filled if
    saved, otherwise deferred to the claim, which refuses the prompt if that leaves it empty).
    """
    seg = _seg()
    prompt = prompt or ""
    out: set[str] = set()
    if seg.SCENE_PLACEHOLDER in prompt:
        out.add(SCENE)
    if seg.MOTION_PLACEHOLDER in prompt and motion_captioning_can_run():
        out.add(MOTION)
    return out


def saved_halves(meta: ImageMeta | None) -> dict[str, str]:
    """The words saved on an image's row, by half. A blank column is not a saved half."""
    out: dict[str, str] = {}
    if meta is None:
        return out
    if (meta.scene_description or "").strip():
        out[SCENE] = meta.scene_description
    if (meta.motion_description or "").strip():
        out[MOTION] = meta.motion_description
    return out


def fill_saved(prompt: str, meta: ImageMeta | None) -> str:
    """Fill each placeholder that has saved words; leave the rest literal.

    The same single non-rescanning pass _resolve_scene uses, so words put in for one half can
    never be read as the other half's placeholder. The text is the row's, verbatim: what the
    lightbox shows is what renders.
    """
    seg = _seg()
    saved = saved_halves(meta)

    def fill(m) -> str:
        half = SCENE if m.group(0) == seg.SCENE_PLACEHOLDER else MOTION
        return saved.get(half, m.group(0))

    return seg._CAPTION_PLACEHOLDERS.sub(fill, prompt)


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
    if not missing and not caption_queue.in_flight(image_uri):
        return fill_saved(prompt, meta), False
    logger.info("Caption hold: holding a segment on %s until its %s %s saved%s",
                image_uri, " and ".join(sorted(missing or needs)),
                "are" if len(missing or needs) > 1 else "is",
                " (a caption of it is running)" if caption_queue.in_flight(image_uri) else "")
    return prompt, True


# ---------------------------------------------------------------------------------------
# Release and failure
# ---------------------------------------------------------------------------------------

async def settle(path: str) -> set[str]:
    """Release every held segment on `path` whose halves are all saved.

    Returns the halves still outstanding across the ones left waiting; empty means nothing
    on this image is held any more (all released, or none were). One short session, rows
    locked so a Retry or Render-without on the same segment cannot interleave.
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
        outstanding: set[str] = set()
        for s in rows:
            missing = needed_halves(s.prompt) - have
            if missing:
                outstanding |= missing
                continue
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


def wait_note(path: str) -> str | None:
    """What the hold on this image is waiting for, in words, with its place in the queue."""
    from app.routes.images import _queue_fields  # deferred: images imports a great deal
    note = _notes.get(path)
    q = _queue_fields(path)
    if q["queue_status"] == "running":
        where = "captioning now"
    elif q["queue_status"] == "queued":
        where = f"{q['queue_position']} of {q['queue_depth']} in the caption queue"
    else:
        where = None
    parts = [p for p in (note, where) if p]
    return "; ".join(parts) if parts else None


async def _hold(path: str) -> None:
    deadline = time.monotonic() + settings.caption_hold_timeout_s
    try:
        while True:
            # 1. Join a caption of this image that is already happening, wherever it came
            #    from. Not a turn of our own behind it: it would describe the image AGAIN and
            #    overwrite the words the modal is about to show.
            if caption_queue.in_flight(path):
                _notes[path] = "waiting for the caption already running for this image"
                while caption_queue.in_flight(path):
                    if time.monotonic() > deadline:
                        await fail(path, _timed_out())
                        return
                    await asyncio.sleep(JOIN_POLL_S)

            outstanding = await settle(path)
            if not outstanding:
                return

            # 2. Nothing is making the words; make them, through the same line the modal
            #    uses. Re-checked at the front: what was ahead of us may have been this image.
            _notes[path] = "waiting for its turn at the captioner"
            try:
                async with caption_queue.turn(path):
                    outstanding = await settle(path)
                    if not outstanding:
                        return
                    _notes[path] = f"captioning the {' and '.join(sorted(outstanding))}"
                    await _caption(path, outstanding)
            except CaptionerBusy as e:
                # The box beside the captioner is rendering. Not a failure -- the modal would
                # be refused the same way -- so wait it out, up to the deadline.
                if time.monotonic() > deadline:
                    await fail(path, f"{_timed_out()} ({e})")
                    return
                _notes[path] = f"the captioner is unavailable: {e}"
                logger.info("Caption hold: %s refused (%s); asking again in %ds",
                            path, e, BUSY_RETRY_S)
                await asyncio.sleep(BUSY_RETRY_S)
                continue

            left = await settle(path)
            if left:
                await fail(path, f"the caption was made but its {' and '.join(sorted(left))} "
                                 "half is still missing")
            return
    except asyncio.CancelledError:
        raise
    except CaptionError as e:
        await fail(path, str(e))
    except Exception as e:  # noqa: BLE001 - whatever it was, a person must hear about it
        logger.exception("Caption hold on %s failed", path)
        await fail(path, f"{type(e).__name__}: {e}")
    finally:
        _notes.pop(path, None)


def _timed_out() -> str:
    return (f"no caption after {settings.caption_hold_timeout_s // 60} minutes of waiting")


async def _caption(path: str, outstanding: set[str]) -> None:
    """Caption `path` and save what it needs. Raises CaptionError / CaptionerBusy.

    A MISSING SCENE is made the way the modal makes it -- the scene and motion pair, from one
    call, written with the modal's own writer -- so a held job's caption is indistinguishable
    from one a person asked for.

    A MISSING MOTION BESIDE A SAVED SCENE makes only the motion paragraph, grounded on the
    saved scene. Regenerating the pair would replace the scene words the person already read,
    which is the opposite of the point.
    """
    from app.routes.app_settings import _get_all_settings
    from app.routes.captions import _caption_base, run_caption_pair
    from app.routes.images import _apply_scene_pair

    try:
        image = await asyncio.to_thread(s3.download_bytes, path)
    except Exception as e:
        raise CaptionError(f"could not read {path}: {e}") from e

    # A short session for everything the captioner call needs from the database, closed
    # before the call. interactive=True: route it exactly as the modal's would be routed --
    # the primary when the box beside it is idle, the fallback or a refusal while it renders.
    async with async_session() as db:
        base = await _caption_base(db, interactive=True)
        cfg = await _get_all_settings(db)
        meta = await db.get(ImageMeta, path)
        saved_scene = (meta.scene_description or "").strip() if meta else ""

    if SCENE in outstanding or not saved_scene:
        pair = await run_caption_pair(image, base, cfg)
        if not pair.scene.strip():
            raise CaptionError("the captioner returned nothing for this image")
        async with async_session() as db:
            meta = await db.get(ImageMeta, path)
            if meta is None:
                meta = ImageMeta(path=path)
                db.add(meta)
            _apply_scene_pair(meta, pair)
            await db.commit()
        logger.info("Caption hold: described %s (scene %d words, motion %s)", path,
                    len(pair.scene.split()),
                    f"{len(pair.motion.split())} words" if pair.motion
                    else (f"failed: {pair.motion_error}" if pair.motion_error else "off"))
        if MOTION in outstanding and not pair.motion:
            raise CaptionError(
                f"the scene was described but the motion caption failed: "
                f"{pair.motion_error or 'the captioner returned no motion paragraph'}")
        return

    motion, instruction = await describe_motion(
        image, saved_scene, style=cfg.get("motion_style", ""),
        custom=cfg.get("motion_instruction", ""), base_url=base)
    motion = (motion or "").strip()
    if not motion:
        raise CaptionError("the captioner returned no motion paragraph")
    async with async_session() as db:
        meta = await db.get(ImageMeta, path)
        if meta is None:  # deleted in the meantime; the scene it was grounded on went too
            raise CaptionError(f"{path} lost its saved description while captioning")
        meta.motion_description = motion
        meta.motion_instruction = instruction
        meta.motion_described_at = datetime.now(timezone.utc)
        await db.commit()
    logger.info("Caption hold: added the motion paragraph to %s (%d words)",
                path, len(motion.split()))


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
    for path in paths:
        ensure(path)
    return len(paths)


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
