"""console#577: a recipe queued before its caption lands must never render empty.

What happened: the console deleted the unfilled placeholder before it submitted, so the caption
hold (console#562) never saw it, the segment went out PENDING, and the claim handed the GPU
"k3lly2026, woman," -- a render with no prompt to speak of. (It was a Motion recipe's
<MOTION> then; that half was removed in wanly-api#481, and a legacy <MOTION> is now stripped
-- the guards below are what keep a stripped-to-the-trigger prompt from rendering.)

Two layers, one class each:

  * THE HOLD, end to end in shape: a "<trigger>, <SCENE>" prompt is held, is not claimable
    while held, is released once the scene is saved, and the claim hands out the saved words
    VERBATIM.
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

from app import caption_hold, caption_tickets
from app import caption_queue as caption_queue_module
from app.auth import get_current_user, verify_api_key
from app.config import settings
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import ImageMeta, Job, Segment, User

IMG = f"s3://{settings.s3_images_bucket}/2026-09-30/00577.png"
SCENE_WORDS = "a woman in a green sweater sitting on the edge of a bed, looking at the viewer"

#: A recipe whose words all come from the start frame's scene caption.
SUBMITTED = "k3lly2026, woman, <SCENE>"
#: A Motion recipe from before wanly-api#481: once <MOTION> is stripped, only the trigger.
LEGACY_MOTION = "k3lly2026, woman, <MOTION>"
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
    caption_hold._waiters.clear()
    caption_hold._notes.clear()
    caption_tickets.reset()
    yield
    caption_hold._waiters.clear()
    caption_tickets.reset()


@pytest.fixture
def shared_session(db, monkeypatch):
    """The hold's short sessions are the test's session, so everything rolls back."""
    @asynccontextmanager
    async def _session():
        yield db
    monkeypatch.setattr(caption_hold, "async_session", _session)
    monkeypatch.setattr(caption_tickets, "async_session", _session)
    return db


@pytest.fixture
def ensured(monkeypatch):
    """Record the routes' ensure() instead of starting a waiter mid-request."""
    calls: list[str] = []
    monkeypatch.setattr(caption_hold, "ensure", lambda path: calls.append(path))
    return calls


@pytest.fixture
def scene_captioner(monkeypatch):
    """The captioner, for the hold's own scene caption."""
    from tests.caption_fakes import ScenePair, drive_scene_from
    pair = AsyncMock(return_value=ScenePair(scene=SCENE_WORDS, scene_instruction="i-scene"))
    drive_scene_from(monkeypatch, pair)
    monkeypatch.setattr("app.routes.captions._caption_base", AsyncMock(return_value="http://c"))
    monkeypatch.setattr(caption_hold.s3, "download_bytes", lambda uri: b"png-bytes")
    return pair


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


async def _meta(db, *, scene=SCENE_WORDS) -> ImageMeta:
    meta = await db.get(ImageMeta, IMG)
    if meta is None:
        meta = ImageMeta(path=IMG)
        db.add(meta)
    meta.scene_description = scene
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
class TestTheSceneRecipeWaitsForItsWords:
    async def test_held_released_and_rendered_with_the_saved_words_verbatim(
            self, db, shared_session, ensured):
        # The frame's scene is still being written.
        user = await _user(db)

        resp = await _submit_job(db, user, SUBMITTED)
        assert resp.status_code == 201, resp.text
        seg = await _only_segment(db, resp.json()["id"])
        assert seg.status == SegmentStatus.AWAITING_CAPTION
        assert seg.prompt == SUBMITTED
        assert ensured == [IMG]

        # Not claimable while it waits -- this is the render that used to go out empty.
        assert await _claim(db) is None

        # The scene lands (the modal's caption, the dialog's, the hold's own -- the release
        # does not care which), and the hold releases the segment.
        await _meta(db)
        assert await caption_hold.settle(IMG) == set()

        body = await _claim(db)
        assert body is not None
        assert body["id"] == str(seg.id)
        assert body["prompt"] == f"k3lly2026, woman, {SCENE_WORDS}"

    async def test_the_holds_own_caption_fills_it(
            self, db, shared_session, ensured, scene_captioner):
        # Nobody else is captioning the frame: the hold's waiter makes the scene itself and
        # releases the segment with exactly those words.
        user = await _user(db)
        resp = await _submit_job(db, user, SUBMITTED)
        assert resp.status_code == 201, resp.text

        await _REAL_ENSURE(IMG)

        scene_captioner.assert_awaited_once()
        assert (await db.get(ImageMeta, IMG)).scene_description == SCENE_WORDS
        body = await _claim(db)
        assert body["prompt"] == f"k3lly2026, woman, {SCENE_WORDS}"

    async def test_a_scene_prompt_gates_on_the_scene_and_motion_never_gates(self):
        assert caption_hold.needed_halves(SUBMITTED) == {"scene"}
        assert caption_hold.needed_halves(LEGACY_MOTION) == set()


# --- the submit guard -------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSubmitRefusesAnEmptyPrompt:
    @pytest.mark.parametrize("prompt", ["", "   ", "<MOTION>", "<SCENE>, <MOTION>",
                                        " , <motion></motion>, ", "<SCENE>"])
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

    async def test_a_legacy_motion_recipe_is_422_and_says_why(self, db, ensured):
        # Once <MOTION> is stripped (wanly-api#481) only the trigger phrase is left.
        user = await _user(db)
        resp = await _submit_job(db, user, LEGACY_MOTION)
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert "trigger phrase" in detail and "<MOTION> is no longer supported" in detail

    @pytest.mark.parametrize("prompt", ["<MOTION>", "", TRIGGER_ONLY, LEGACY_MOTION])
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
    async def test_a_legacy_motion_that_leaves_the_trigger_is_caption_failed(self, db):
        # A PENDING segment queued before wanly-api#481 can still carry <MOTION>. The claim
        # strips it (persisted), and what is left is only the trigger phrase.
        await _meta(db)
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, LEGACY_MOTION)

        assert await _claim(db) is None

        await db.refresh(seg)
        await db.refresh(job)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert "trigger phrase" in seg.error_message
        # Stripped for good: a Retry can never hold for it; nothing was assigned a worker.
        assert "<MOTION>" not in seg.prompt
        assert seg.prompt == "k3lly2026, woman"
        assert seg.worker_id is None and seg.claimed_at is None
        assert job.status == JobStatus.PENDING

    async def test_a_legacy_motion_beside_real_words_is_stripped_and_handed_out(self, db):
        await _meta(db)
        user = await _user(db)
        seg = await _segment(db, await _job(db, user),
                             "k3lly2026, woman, <SCENE>, <MOTION>, she waves")
        body = await _claim(db)
        assert body["id"] == str(seg.id)
        assert body["prompt"] == f"k3lly2026, woman, {SCENE_WORDS}, she waves"
        assert "<MOTION>" not in body["prompt"]

    async def test_a_filled_legacy_motion_region_is_stripped_words_and_all(self, db):
        user = await _user(db)
        seg = await _segment(db, await _job(db, user),
                             "k3lly2026, woman, she waves, <motion>old paragraph</motion>")
        body = await _claim(db)
        assert body["id"] == str(seg.id)
        assert body["prompt"] == "k3lly2026, woman, she waves"

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
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, SUBMITTED, status=SegmentStatus.CAPTION_FAILED)
        async with _client(db, user) as c:
            resp = await c.post(f"/segments/{seg.id}/caption/skip")
        assert resp.status_code == 422, resp.text
        assert "<SCENE>" in resp.json()["detail"]
        await db.refresh(seg)
        assert seg.status == SegmentStatus.CAPTION_FAILED
        assert seg.prompt == SUBMITTED

    async def test_it_still_works_when_something_is_left(self, db, ensured):
        user = await _user(db)
        job = await _job(db, user)
        seg = await _segment(db, job, "k3lly2026, woman, she waves, <SCENE>, <MOTION>",
                             status=SegmentStatus.CAPTION_FAILED)
        async with _client(db, user) as c:
            resp = await c.post(f"/segments/{seg.id}/caption/skip")
        assert resp.status_code == 200, resp.text
        assert resp.json()["prompt"] == "k3lly2026, woman, she waves"
        assert resp.json()["status"] == SegmentStatus.PENDING
