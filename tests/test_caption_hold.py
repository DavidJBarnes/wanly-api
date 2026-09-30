"""The caption hold (console#562): a job never renders until its start image's captions exist.

What this file protects, one class each:

  * a held segment is NEVER claimable -- awaiting or failed;
  * the submit decides without captioning, and returns at once;
  * release fills EXACTLY the saved words, verbatim;
  * single-flight: a job waiting on an image joins the caption already running for it (the
    modal's, the dialog's, another job's) instead of starting a second one;
  * failure is loud -- caption_failed with the reason -- and Retry / Render without are the
    only ways out;
  * the sweep re-creates the waiters after a restart;
  * with motion captioning off only <SCENE> gates;
  * continuations whose frame does not exist yet are untouched.

The captioner is a mock that records, so "was not called" is an assertion, not a hope.
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app import caption_hold
from app.auth import get_current_user, verify_api_key
from app import caption_queue as caption_queue_module
from app.config import settings
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.joycaption import CaptionError, CaptionerBusy
from app.main import app
from app.models import ImageMeta, Job, Segment, User
from app.routes.captions import ScenePair

IMG = f"s3://{settings.s3_images_bucket}/2026-09-30/00042.png"
OTHER = f"s3://{settings.s3_images_bucket}/2026-09-30/00043.png"
SCENE_WORDS = "a woman in a red dress sitting on a sofa, looking at the viewer"
MOTION_WORDS = "she leans forward slowly, her hair swaying as she reaches toward the camera"
BOTH = "k3lly, <SCENE>, <MOTION>"
SCENE_ONLY = "k3lly, <SCENE>, she smiles"


# --- fixtures ------------------------------------------------------------------------------

@pytest.fixture
def shared_session(db, monkeypatch):
    """Point the module's short sessions at the test's session, so everything rolls back."""
    @asynccontextmanager
    async def _session():
        yield db
    monkeypatch.setattr(caption_hold, "async_session", _session)
    return db


class _Q:
    """The queue each test uses: a fresh one, so no test's asyncio.Lock is bound to another
    test's event loop (pytest-asyncio gives each test its own)."""
    def turn(self, path):
        return caption_hold.caption_queue.turn(path)


caption_queue = _Q()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    fresh = caption_queue_module.CaptionQueue()
    monkeypatch.setattr(caption_queue_module, "queue", fresh)
    monkeypatch.setattr(caption_hold, "caption_queue", fresh)
    monkeypatch.setattr(caption_hold, "JOIN_POLL_S", 0.01)
    monkeypatch.setattr(caption_hold, "BUSY_RETRY_S", 0.01)
    monkeypatch.setattr(settings, "motion_caption_enabled", True)
    caption_hold._waiters.clear()
    caption_hold._notes.clear()
    yield
    caption_hold._waiters.clear()


@pytest.fixture
def captioner(monkeypatch):
    """The captioner and everything around it, mocked. `.pair` and `.motion` record calls."""
    class C:
        pair = AsyncMock(return_value=ScenePair(
            scene=SCENE_WORDS, scene_instruction="i-scene",
            motion=MOTION_WORDS, motion_instruction="i-motion"))
        motion = AsyncMock(return_value=(MOTION_WORDS, "i-motion"))
    monkeypatch.setattr("app.routes.captions.run_caption_pair", C.pair)
    monkeypatch.setattr(caption_hold, "describe_motion", C.motion)
    monkeypatch.setattr("app.routes.captions._caption_base", AsyncMock(return_value="http://c"))
    monkeypatch.setattr(caption_hold.s3, "download_bytes", lambda uri: b"png-bytes")
    return C


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user=None, *, starting_image=IMG, status=JobStatus.PENDING) -> Job:
    user = user or await _user(db)
    job = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=7,
              status=status, starting_image=starting_image)
    db.add(job)
    await db.flush()
    return job


async def _held(db, job, *, prompt=BOTH, index=0, start_image=None,
                status=SegmentStatus.AWAITING_CAPTION) -> Segment:
    seg = Segment(job_id=job.id, index=index, prompt=prompt, status=status,
                  start_image=start_image)
    db.add(seg)
    await db.flush()
    return seg


async def _meta(db, path=IMG, scene=SCENE_WORDS, motion=MOTION_WORDS) -> ImageMeta:
    meta = await db.get(ImageMeta, path)
    if meta is None:
        meta = ImageMeta(path=path)
        db.add(meta)
    meta.scene_description = scene
    meta.motion_description = motion
    await db.flush()
    return meta


async def _fresh(db, obj):
    await db.refresh(obj)
    return obj


# --- pure rules ------------------------------------------------------------------------------

class TestWhatGates:
    def test_both_halves_gate_when_motion_is_on(self):
        assert caption_hold.needed_halves(BOTH) == {"scene", "motion"}

    def test_motion_off_leaves_only_the_scene_gating(self, monkeypatch):
        # The kill-switch (MOTION_CAPTION_ENABLED=false, for a captioner that cannot do the
        # motion half). Waiting for a paragraph no caption will ever produce would be waiting
        # forever.
        monkeypatch.setattr(settings, "motion_caption_enabled", False)
        assert caption_hold.needed_halves(BOTH) == {"scene"}
        assert caption_hold.needed_halves("k3lly, <MOTION>") == set()

    def test_a_filled_prompt_needs_nothing(self):
        assert caption_hold.needed_halves("k3lly, a woman, she smiles") == set()

    def test_hold_image_is_the_segments_own_then_the_jobs_for_segment_zero(self):
        assert caption_hold.hold_image("s3://x/a.png", 3, "s3://x/j.png") == "s3://x/a.png"
        assert caption_hold.hold_image(None, 0, "s3://x/j.png") == "s3://x/j.png"
        # A continuation with no start image continues from a frame that does not exist yet.
        assert caption_hold.hold_image(None, 2, "s3://x/j.png") is None


class TestFillIsVerbatim:
    def test_it_fills_exactly_the_saved_words(self):
        meta = ImageMeta(path=IMG, scene_description=SCENE_WORDS,
                         motion_description=MOTION_WORDS)
        assert caption_hold.fill_saved(BOTH, meta) == f"k3lly, {SCENE_WORDS}, {MOTION_WORDS}"

    def test_a_caption_containing_the_other_placeholder_stays_words(self):
        # Single pass: the scene's literal "<MOTION>" must not become a second substitution.
        meta = ImageMeta(path=IMG, scene_description="a sign reading <MOTION>",
                         motion_description=MOTION_WORDS)
        out = caption_hold.fill_saved(BOTH, meta)
        assert out == f"k3lly, a sign reading <MOTION>, {MOTION_WORDS}"

    def test_a_missing_half_stays_literal(self):
        meta = ImageMeta(path=IMG, scene_description=SCENE_WORDS)
        assert caption_hold.fill_saved(BOTH, meta) == f"k3lly, {SCENE_WORDS}, <MOTION>"


# --- the submit decision ---------------------------------------------------------------------

@pytest.mark.asyncio
class TestGate:
    async def test_no_image_is_not_ours_so_continuations_keep_their_deferral(self, db):
        assert await caption_hold.gate(db, BOTH, None) is None

    async def test_a_frame_outside_our_buckets_is_not_ours(self, db):
        assert await caption_hold.gate(db, BOTH, "s3://somebody-else/x.png") is None

    async def test_a_prompt_without_placeholders_is_not_ours(self, db):
        assert await caption_hold.gate(db, "k3lly, a woman", IMG) is None

    async def test_saved_words_fill_at_once_and_nothing_is_held(self, db):
        await _meta(db)
        prompt, held = await caption_hold.gate(db, BOTH, IMG)
        assert held is False
        assert prompt == f"k3lly, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_a_missing_scene_holds(self, db):
        prompt, held = await caption_hold.gate(db, BOTH, IMG)
        assert held is True and prompt == BOTH

    async def test_a_missing_motion_holds_when_motion_is_on(self, db):
        await _meta(db, motion=None)
        _, held = await caption_hold.gate(db, BOTH, IMG)
        assert held is True

    async def test_motion_off_releases_on_the_scene_alone(self, db, monkeypatch):
        monkeypatch.setattr(settings, "motion_caption_enabled", False)
        await _meta(db, motion=None)
        prompt, held = await caption_hold.gate(db, BOTH, IMG)
        assert held is False
        # <MOTION> keeps its pre-#562 behaviour: literal here, the claim's business.
        assert prompt == f"k3lly, {SCENE_WORDS}, <MOTION>"

    async def test_a_caption_in_flight_holds_even_over_saved_words(self, db):
        # A re-roll running in the modal: the words about to land are the ones on screen.
        await _meta(db)
        entered, release = asyncio.Event(), asyncio.Event()

        async def modal():
            async with caption_queue.turn(IMG):
                entered.set()
                await release.wait()
        t = asyncio.create_task(modal())
        await entered.wait()
        try:
            _, held = await caption_hold.gate(db, BOTH, IMG)
            assert held is True
        finally:
            release.set()
            await t


# --- never claimable -----------------------------------------------------------------------

async def _claim(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment)):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await c.get("/segments/next", params={
                "worker_id": str(uuid.uuid4()), "worker_name": "3090.zero", "kind": "gpu"})
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestNeverClaimable:
    @pytest.mark.parametrize("held", [SegmentStatus.AWAITING_CAPTION,
                                      SegmentStatus.CAPTION_FAILED])
    async def test_a_held_segment_is_never_handed_out(self, db, held):
        job = await _job(db)
        await _held(db, job, status=held)
        resp = await _claim(db)
        assert resp.status_code == 200, resp.text
        assert resp.json() is None

    async def test_the_queue_moves_past_a_held_segment(self, db):
        held_job = await _job(db)
        await _held(db, held_job)
        ready_job = await _job(db)
        ready = await _held(db, ready_job, prompt="k3lly, a woman",
                            status=SegmentStatus.PENDING)
        body = (await _claim(db)).json()
        assert body["id"] == str(ready.id)


# --- release -------------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSettle:
    async def test_release_fills_exactly_the_saved_text(self, db, shared_session):
        job = await _job(db)
        seg = await _held(db, job)
        await _meta(db)
        assert await caption_hold.settle(IMG) == set()
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.PENDING
        assert seg.prompt == f"k3lly, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_it_keeps_waiting_while_a_half_is_missing(self, db, shared_session):
        job = await _job(db)
        seg = await _held(db, job)
        await _meta(db, motion=None)
        assert await caption_hold.settle(IMG) == {"motion"}
        assert (await _fresh(db, seg)).status == SegmentStatus.AWAITING_CAPTION

    async def test_a_segment_on_another_image_is_not_touched(self, db, shared_session):
        job = await _job(db, starting_image=OTHER)
        seg = await _held(db, job)
        await _meta(db)
        await caption_hold.settle(IMG)
        assert (await _fresh(db, seg)).status == SegmentStatus.AWAITING_CAPTION

    async def test_its_own_start_image_beats_the_jobs(self, db, shared_session):
        job = await _job(db, starting_image=OTHER)
        seg = await _held(db, job, index=1, start_image=IMG, prompt=SCENE_ONLY)
        await _meta(db)
        await caption_hold.settle(IMG)
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.PENDING
        assert seg.prompt == f"k3lly, {SCENE_WORDS}, she smiles"


# --- the waiter ----------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSingleFlight:
    async def test_a_job_joins_the_modals_caption_instead_of_making_its_own(
            self, db, shared_session, captioner):
        """The user's flow: caption in the modal, Use as Starting Image, submit, carry on."""
        job = await _job(db)
        seg = await _held(db, job)
        entered, finish = asyncio.Event(), asyncio.Event()

        async def modal():
            # POST /images/scene: a turn in the queue, then the pair written on the row.
            async with caption_queue.turn(IMG):
                entered.set()
                await finish.wait()
                await _meta(db, scene="the modal's words", motion="the modal's motion")
        m = asyncio.create_task(modal())
        await entered.wait()

        waiter = caption_hold.ensure(IMG)
        await asyncio.sleep(0.05)
        assert (await _fresh(db, seg)).status == SegmentStatus.AWAITING_CAPTION
        finish.set()
        await asyncio.wait_for(asyncio.gather(m, waiter), 5)

        captioner.pair.assert_not_called()
        captioner.motion.assert_not_called()
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.PENDING
        assert seg.prompt == "k3lly, the modal's words, the modal's motion"

    async def test_two_jobs_on_one_image_share_one_waiter_and_one_caption(
            self, db, shared_session, captioner):
        a = await _held(db, await _job(db))
        b = await _held(db, await _job(db))
        first, second = caption_hold.ensure(IMG), caption_hold.ensure(IMG)
        assert first is second
        await asyncio.wait_for(first, 5)
        assert captioner.pair.await_count == 1
        for seg in (a, b):
            seg = await _fresh(db, seg)
            assert seg.status == SegmentStatus.PENDING
            assert seg.prompt == f"k3lly, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_with_nothing_in_flight_it_captions_through_the_queue_and_saves(
            self, db, shared_session, captioner):
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == SCENE_WORDS
        assert meta.motion_description == MOTION_WORDS
        assert (await _fresh(db, seg)).prompt == f"k3lly, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_words_landing_while_it_queued_are_used_not_redone(
            self, db, shared_session, captioner):
        """Whatever was ahead of it in line may have been this image."""
        seg = await _held(db, await _job(db))
        entered, finish = asyncio.Event(), asyncio.Event()

        async def other_image_then_this_one():
            async with caption_queue.turn(OTHER):
                entered.set()
                await finish.wait()
                await _meta(db)  # somebody else described IMG meanwhile
        t = asyncio.create_task(other_image_then_this_one())
        await entered.wait()
        waiter = caption_hold.ensure(IMG)
        await asyncio.sleep(0.05)
        finish.set()
        await asyncio.wait_for(asyncio.gather(t, waiter), 5)
        captioner.pair.assert_not_called()
        assert (await _fresh(db, seg)).status == SegmentStatus.PENDING

    async def test_a_saved_scene_is_kept_and_only_the_motion_is_made(
            self, db, shared_session, captioner):
        # Regenerating the pair would replace the scene the person already read.
        await _meta(db, scene="the words she read", motion=None)
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        captioner.pair.assert_not_called()
        assert captioner.motion.await_args.args[1] == "the words she read"
        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == "the words she read"
        assert (await _fresh(db, seg)).prompt == f"k3lly, the words she read, {MOTION_WORDS}"


@pytest.mark.asyncio
class TestFailureIsLoud:
    async def test_a_caption_error_fails_the_segment_with_the_reason(
            self, db, shared_session, captioner):
        captioner.pair.side_effect = CaptionError("captioner unreachable at http://c")
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "captioner unreachable" in seg.error_message
        assert seg.prompt == BOTH  # nothing dropped

    async def test_a_failed_motion_half_fails_it_and_keeps_the_scene(
            self, db, shared_session, captioner):
        captioner.pair.return_value = ScenePair(
            scene=SCENE_WORDS, scene_instruction="i", motion_error="timed out")
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "motion caption failed: timed out" in seg.error_message
        assert (await db.get(ImageMeta, IMG)).scene_description == SCENE_WORDS

    async def test_an_unreadable_image_fails_it(self, db, shared_session, captioner,
                                                monkeypatch):
        def gone(uri):
            raise FileNotFoundError(uri)
        monkeypatch.setattr(caption_hold.s3, "download_bytes", gone)
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "could not read" in seg.error_message

    async def test_a_busy_box_is_waited_out_then_times_out(
            self, db, shared_session, captioner, monkeypatch):
        monkeypatch.setattr("app.routes.captions._caption_base",
                            AsyncMock(side_effect=CaptionerBusy("3090.zero is rendering")))
        monkeypatch.setattr(settings, "caption_hold_timeout_s", 0)
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "no caption after" in seg.error_message
        assert "3090.zero is rendering" in seg.error_message

    async def test_a_busy_box_that_frees_up_is_captioned(
            self, db, shared_session, captioner, monkeypatch):
        base = AsyncMock(side_effect=[CaptionerBusy("rendering"), "http://c"])
        monkeypatch.setattr("app.routes.captions._caption_base", base)
        seg = await _held(db, await _job(db))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        assert (await _fresh(db, seg)).status == SegmentStatus.PENDING
        assert base.await_count == 2


# --- restart -------------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRestartSweep:
    async def test_a_held_segment_with_no_waiter_is_resumed_and_released(
            self, db, shared_session, captioner):
        # What a restart leaves: the row, and nothing in memory watching it. The words were
        # saved while the API was down.
        seg = await _held(db, await _job(db))
        await _meta(db)
        assert caption_hold._waiters == {}
        assert await caption_hold.sweep() == 1
        await asyncio.wait_for(caption_hold._waiters[IMG], 5)
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.PENDING
        captioner.pair.assert_not_called()

    async def test_a_resumed_hold_with_no_words_captions(self, db, shared_session, captioner):
        seg = await _held(db, await _job(db))
        await caption_hold.sweep()
        await asyncio.wait_for(caption_hold._waiters[IMG], 5)
        assert (await _fresh(db, seg)).status == SegmentStatus.PENDING
        assert captioner.pair.await_count == 1

    async def test_a_held_row_with_nothing_to_caption_fails_rather_than_waits(
            self, db, shared_session):
        seg = await _held(db, await _job(db, starting_image=None), index=0)
        await caption_hold.sweep()
        seg = await _fresh(db, seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "no start image" in seg.error_message

    async def test_failed_segments_are_not_retried_behind_the_persons_back(
            self, db, shared_session, captioner):
        await _held(db, await _job(db), status=SegmentStatus.CAPTION_FAILED)
        assert await caption_hold.sweep() == 0
        captioner.pair.assert_not_called()


# --- the routes ----------------------------------------------------------------------------

@asynccontextmanager
async def _client(db, user):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def ensured(monkeypatch):
    """Record ensure() instead of starting a waiter: the routes' job is only to ask."""
    calls: list[str] = []
    monkeypatch.setattr(caption_hold, "ensure", lambda path: calls.append(path))
    return calls


@pytest.fixture
def no_live_caption(monkeypatch):
    """Any live caption on the submit path fails the test."""
    boom = AsyncMock(side_effect=AssertionError("the submit captioned synchronously"))
    monkeypatch.setattr("app.routes.segments.caption_image_bytes", boom)
    return boom


@pytest.mark.asyncio
class TestSubmit:
    async def test_new_job_is_held_and_returns_without_captioning(
            self, db, ensured, no_live_caption):
        user = await _user(db)
        data = {"name": "j", "width": 832, "height": 1216, "fps": 24,
                "starting_image_uri": IMG, "first_segment": {"prompt": BOTH}}
        async with _client(db, user) as c:
            resp = await c.post("/jobs", data={"data": json.dumps(data)})
        assert resp.status_code in (200, 201), resp.text
        seg = (await db.execute(
            Segment.__table__.select().where(Segment.job_id == uuid.UUID(resp.json()["id"]))
        )).one()
        assert seg.status == SegmentStatus.AWAITING_CAPTION
        assert seg.prompt == BOTH
        assert ensured == [IMG]

    async def test_new_job_with_saved_words_is_filled_and_pending(self, db, ensured):
        await _meta(db)
        user = await _user(db)
        data = {"name": "j", "width": 832, "height": 1216, "fps": 24,
                "starting_image_uri": IMG, "first_segment": {"prompt": BOTH}}
        async with _client(db, user) as c:
            resp = await c.post("/jobs", data={"data": json.dumps(data)})
        seg = (await db.execute(
            Segment.__table__.select().where(Segment.job_id == uuid.UUID(resp.json()["id"]))
        )).one()
        assert seg.status == SegmentStatus.PENDING
        assert seg.prompt == f"k3lly, {SCENE_WORDS}, {MOTION_WORDS}"
        assert ensured == []

    async def test_next_segment_with_a_start_image_is_held_not_captioned(
            self, db, ensured, no_live_caption):
        user = await _user(db)
        job = await _job(db, user, status=JobStatus.AWAITING)
        async with _client(db, user) as c:
            resp = await c.post(f"/jobs/{job.id}/segments",
                                json={"prompt": SCENE_ONLY, "start_image": OTHER})
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == SegmentStatus.AWAITING_CAPTION
        assert ensured == [OTHER]
        no_live_caption.assert_not_called()

    async def test_a_continuation_keeps_its_deferral(self, db, ensured, no_live_caption):
        # No start image: the frame it continues from does not exist until the previous
        # segment renders. Held on nothing, placeholder kept for the claim, as before.
        user = await _user(db)
        job = await _job(db, user, status=JobStatus.AWAITING)
        await _held(db, job, prompt="p", status=SegmentStatus.COMPLETED)
        async with _client(db, user) as c:
            resp = await c.post(f"/jobs/{job.id}/segments", json={"prompt": BOTH})
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == SegmentStatus.PENDING
        assert resp.json()["prompt"] == BOTH
        assert ensured == []

    async def test_job_list_and_detail_say_why_it_is_not_starting(self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        await _held(db, job)
        async with _client(db, user) as c:
            listed = (await c.get("/jobs")).json()["items"]
            detail = (await c.get(f"/jobs/{job.id}")).json()
        assert [j["caption_hold"] for j in listed if j["id"] == str(job.id)] == [
            "awaiting_caption"]
        assert detail["caption_hold"] == "awaiting_caption"
        assert detail["segments"][0]["caption_image"] == IMG


@pytest.mark.asyncio
class TestTheTwoWaysOut:
    async def test_render_without_fills_what_is_saved_and_drops_the_rest(self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _held(db, job, status=SegmentStatus.CAPTION_FAILED)
        await _meta(db, motion=None)
        async with _client(db, user) as c:
            resp = await c.post(f"/segments/{seg.id}/caption/skip")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == SegmentStatus.PENDING
        assert body["prompt"] == f"k3lly, {SCENE_WORDS}"
        assert body["error_message"] is None

    async def test_render_without_leaves_no_placeholder_for_the_claim_to_caption(
            self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _held(db, job, status=SegmentStatus.CAPTION_FAILED)
        async with _client(db, user) as c:
            body = (await c.post(f"/segments/{seg.id}/caption/skip")).json()
        assert "<SCENE>" not in body["prompt"] and "<MOTION>" not in body["prompt"]

    async def test_retry_goes_back_to_waiting_and_asks_for_a_caption(self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _held(db, job, status=SegmentStatus.CAPTION_FAILED)
        seg.error_message = "Caption failed: boom"
        await db.flush()
        async with _client(db, user) as c:
            body = (await c.post(f"/segments/{seg.id}/caption/retry")).json()
        assert body["status"] == SegmentStatus.AWAITING_CAPTION
        assert body["error_message"] is None
        assert ensured == [IMG]

    async def test_neither_applies_to_a_segment_that_is_not_held(self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _held(db, job, status=SegmentStatus.PENDING)
        async with _client(db, user) as c:
            assert (await c.post(f"/segments/{seg.id}/caption/retry")).status_code == 400
            assert (await c.post(f"/segments/{seg.id}/caption/skip")).status_code == 400

    async def test_a_held_segment_can_be_cancelled(self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _held(db, job)
        async with _client(db, user) as c:
            resp = await c.post(f"/segments/{seg.id}/cancel")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == SegmentStatus.FAILED
