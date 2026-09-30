"""console#577: a Motion recipe queued before its caption lands must never render empty.

What happened: the console deleted the unfilled <MOTION> before it submitted, so the caption
hold (console#562) never saw a placeholder, the segment went out PENDING, and the claim handed
the GPU "k3lly2026, woman," -- a render with no prompt to speak of.

Two layers, one class each:

  * THE HOLD, end to end in shape. The prompt is exactly what the console's submitPrompt now
    sends (pinned by the console's captionRegion.test.ts, which asserts this same string): the
    job is held, is not claimable while held, is released once the motion paragraph is saved,
    and the claim hands out the saved words VERBATIM.
  * THE GUARDS. Whatever the client does: a prompt that is blank once the placeholders are
    removed is a 422 at submit, and a prompt that resolves to blank or only the trigger phrase
    is never handed out -- it goes to caption_failed with the reason, and Render without
    refuses to release it while it is still empty.
"""
import json
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app import caption_hold
from app import caption_queue as caption_queue_module
from app.auth import get_current_user, verify_api_key
from app.config import settings
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import ImageMeta, Job, Segment, User

IMG = f"s3://{settings.s3_images_bucket}/2026-09-30/00577.png"
SCENE_WORDS = "a woman in a green sweater sitting on the edge of a bed, looking at the viewer"
MOTION_WORDS = ("she slowly leans back onto her elbows, her hair falling off one shoulder, "
                "and smiles as the camera drifts closer")

#: What the console's submitPrompt sends for a Motion recipe whose caption has not landed.
#: The console test "sends a Motion recipe's prompt exactly as the API's hold test posts it"
#: pins the same string -- change both or neither.
SUBMITTED = "k3lly2026, woman, <MOTION>"
TRIGGER_ONLY = "k3lly2026, woman,"
RECIPE = {"characters": [{"name": "Kelly 577", "trigger": "k3lly2026", "gender": "woman"}]}

_REAL_ENSURE = caption_hold.ensure


# --- fixtures and helpers ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_hold_state(monkeypatch):
    fresh = caption_queue_module.CaptionQueue()
    monkeypatch.setattr(caption_queue_module, "queue", fresh)
    monkeypatch.setattr(caption_hold, "caption_queue", fresh)
    monkeypatch.setattr(caption_hold, "JOIN_POLL_S", 0.01)
    monkeypatch.setattr(settings, "motion_caption_enabled", True)
    caption_hold._waiters.clear()
    caption_hold._notes.clear()
    yield
    caption_hold._waiters.clear()


@pytest.fixture
def shared_session(db, monkeypatch):
    """The hold's short sessions are the test's session, so everything rolls back."""
    @asynccontextmanager
    async def _session():
        yield db
    monkeypatch.setattr(caption_hold, "async_session", _session)
    return db


@pytest.fixture
def ensured(monkeypatch):
    """Record the routes' ensure() instead of starting a waiter mid-request."""
    calls: list[str] = []
    monkeypatch.setattr(caption_hold, "ensure", lambda path: calls.append(path))
    return calls


@pytest.fixture
def motion_captioner(monkeypatch):
    """The captioner, for the motion-only path: the scene is saved, the paragraph is not."""
    motion = AsyncMock(return_value=(MOTION_WORDS, "i-motion"))
    pair = AsyncMock(side_effect=AssertionError("the saved scene must not be re-described"))
    monkeypatch.setattr(caption_hold, "describe_motion", motion)
    monkeypatch.setattr("app.routes.captions.run_caption_pair", pair)
    monkeypatch.setattr("app.routes.captions._caption_base", AsyncMock(return_value="http://c"))
    monkeypatch.setattr(caption_hold.s3, "download_bytes", lambda uri: b"png-bytes")
    return motion


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user, *, status=JobStatus.PENDING) -> Job:
    job = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=7,
              status=status, starting_image=IMG)
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, prompt, *, status=SegmentStatus.PENDING, index=0) -> Segment:
    seg = Segment(job_id=job.id, index=index, prompt=prompt, status=status,
                  ltx_recipe=RECIPE)
    db.add(seg)
    await db.flush()
    return seg


async def _meta(db, *, scene=SCENE_WORDS, motion=None) -> ImageMeta:
    meta = await db.get(ImageMeta, IMG)
    if meta is None:
        meta = ImageMeta(path=IMG)
        db.add(meta)
    meta.scene_description = scene
    meta.motion_description = motion
    await db.flush()
    return meta


@asynccontextmanager
async def _client(db, user):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


async def _claim(db):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment)):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/segments/next", params={
                "worker_id": str(uuid.uuid4()), "worker_name": "3090.zero", "kind": "gpu"})
        assert resp.status_code == 200, resp.text
        return resp.json()
    finally:
        app.dependency_overrides.clear()


async def _submit_job(db, user, prompt):
    data = {"name": "j", "width": 832, "height": 1216, "fps": 24,
            "starting_image_uri": IMG,
            "first_segment": {"prompt": prompt, "ltx_recipe": RECIPE}}
    async with _client(db, user) as c:
        return await c.post("/jobs", data={"data": json.dumps(data)})


async def _only_segment(db, job_id) -> Segment:
    return (await db.execute(
        Segment.__table__.select().where(Segment.job_id == uuid.UUID(str(job_id)))
    )).one()


# --- the hold, end to end in shape ------------------------------------------------------------

@pytest.mark.asyncio
class TestTheMotionRecipeWaitsForItsWords:
    async def test_held_released_and_rendered_with_the_saved_words_verbatim(
            self, db, shared_session, ensured):
        # The frame's scene is described; the motion paragraph is still being written.
        await _meta(db, motion=None)
        user = await _user(db)

        resp = await _submit_job(db, user, SUBMITTED)
        assert resp.status_code == 201, resp.text
        seg = await _only_segment(db, resp.json()["id"])
        assert seg.status == SegmentStatus.AWAITING_CAPTION
        assert seg.prompt == SUBMITTED
        assert ensured == [IMG]

        # Not claimable while it waits -- this is the render that used to go out empty.
        assert await _claim(db) is None

        # The paragraph lands (the modal's caption, the dialog's, the hold's own -- the
        # release does not care which), and the hold releases the segment.
        await _meta(db, motion=MOTION_WORDS)
        assert await caption_hold.settle(IMG) == set()

        body = await _claim(db)
        assert body is not None
        assert body["id"] == str(seg.id)
        assert body["prompt"] == f"k3lly2026, woman, {MOTION_WORDS}"
        assert MOTION_WORDS in body["prompt"]

    async def test_the_holds_own_caption_fills_it(
            self, db, shared_session, ensured, motion_captioner):
        # Nobody else is captioning the frame: the hold's waiter makes the paragraph itself,
        # grounded on the saved scene, and releases the segment with exactly those words.
        await _meta(db, motion=None)
        user = await _user(db)
        resp = await _submit_job(db, user, SUBMITTED)
        assert resp.status_code == 201, resp.text

        await _REAL_ENSURE(IMG)

        motion_captioner.assert_awaited_once()
        assert (await db.get(ImageMeta, IMG)).motion_description == MOTION_WORDS
        body = await _claim(db)
        assert body["prompt"] == f"k3lly2026, woman, {MOTION_WORDS}"

    async def test_a_motion_only_prompt_gates_on_motion(self):
        assert caption_hold.needed_halves(SUBMITTED) == {"motion"}


# --- the submit guard -------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSubmitRefusesAnEmptyPrompt:
    @pytest.mark.parametrize("prompt", ["", "   ", "<MOTION>", "<SCENE>, <MOTION>",
                                        " , <motion></motion>, "])
    async def test_new_job_blank_once_placeholders_are_removed_is_422(
            self, db, ensured, prompt):
        user = await _user(db)
        resp = await _submit_job(db, user, prompt)
        assert resp.status_code == 422, resp.text
        assert (await db.execute(
            Job.__table__.select().where(Job.user_id == user.id))).first() is None

    async def test_new_job_trigger_only_with_no_placeholder_is_422(self, db, ensured):
        # What a console that still strips the placeholder sends.
        user = await _user(db)
        resp = await _submit_job(db, user, TRIGGER_ONLY)
        assert resp.status_code == 422, resp.text
        assert "trigger phrase" in resp.json()["detail"]

    async def test_trigger_phrase_plus_placeholder_is_accepted(self, db, ensured):
        user = await _user(db)
        resp = await _submit_job(db, user, SUBMITTED)
        assert resp.status_code == 201, resp.text

    @pytest.mark.parametrize("prompt", ["<MOTION>", "", TRIGGER_ONLY])
    async def test_next_segment_empty_is_422(self, db, ensured, prompt):
        user = await _user(db)
        job = await _job(db, user, status=JobStatus.AWAITING)
        async with _client(db, user) as c:
            resp = await c.post(f"/jobs/{job.id}/segments",
                                json={"prompt": prompt, "start_image": IMG,
                                      "ltx_recipe": RECIPE})
        assert resp.status_code == 422, resp.text


# --- the claim guard --------------------------------------------------------------------------

@pytest.mark.asyncio
class TestClaimNeverHandsOutAnEmptyPrompt:
    async def test_an_unfilled_motion_that_resolves_to_the_trigger_is_caption_failed(
            self, db, monkeypatch):
        # How a PENDING segment can still carry an unfilled <MOTION>: the kill-switch is off
        # (so the hold does not gate on it), or it was queued before this fix. The claim
        # drops the placeholder, and what is left is only the trigger phrase.
        monkeypatch.setattr(settings, "motion_caption_enabled", False)
        await _meta(db, motion=None)
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, SUBMITTED)

        assert await _claim(db) is None

        await db.refresh(seg)
        await db.refresh(job)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "<MOTION>" in seg.error_message and "trigger phrase" in seg.error_message
        # The placeholder survives for a Retry to fill; nothing was assigned a worker.
        assert seg.prompt == SUBMITTED
        assert seg.worker_id is None and seg.claimed_at is None
        assert job.status == JobStatus.PENDING

    @pytest.mark.parametrize("prompt", ["", " , ", TRIGGER_ONLY, "k3lly2026"])
    async def test_blank_or_trigger_only_is_never_handed_out(self, db, prompt):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, prompt)
        assert await _claim(db) is None
        await db.refresh(seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert seg.error_message.startswith("Caption failed:")

    async def test_the_queue_moves_past_it(self, db):
        user = await _user(db)
        empty = await _segment(db, await _job(db, user), TRIGGER_ONLY)
        ready = await _segment(db, await _job(db, user), "k3lly2026, woman, she waves")
        assert await _claim(db) is None
        body = await _claim(db)
        assert body["id"] == str(ready.id)
        await db.refresh(empty)
        assert empty.status == SegmentStatus.CAPTION_FAILED

    async def test_a_real_prompt_is_handed_out(self, db):
        user = await _user(db)
        seg = await _segment(db, await _job(db, user), "k3lly2026, woman, she waves")
        body = await _claim(db)
        assert body["id"] == str(seg.id)
        assert body["prompt"] == "k3lly2026, woman, she waves"


# --- render without ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRenderWithoutRefusesAnEmptyPrompt:
    async def test_dropping_the_only_half_is_refused(self, db, ensured):
        await _meta(db, motion=None)
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, SUBMITTED, status=SegmentStatus.CAPTION_FAILED)
        async with _client(db, user) as c:
            resp = await c.post(f"/segments/{seg.id}/caption/skip")
        assert resp.status_code == 422, resp.text
        assert "<MOTION>" in resp.json()["detail"]
        await db.refresh(seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert seg.prompt == SUBMITTED

    async def test_it_still_works_when_something_is_left(self, db, ensured):
        await _meta(db, motion=None)
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, "k3lly2026, woman, she waves, <MOTION>",
                             status=SegmentStatus.CAPTION_FAILED)
        async with _client(db, user) as c:
            resp = await c.post(f"/segments/{seg.id}/caption/skip")
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == "k3lly2026, woman, she waves"
        assert resp.json()["status"] == SegmentStatus.PENDING
