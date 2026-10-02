"""Caption tickets (console#564) and what a held job can see of them (console#562 follow-up).

What this file protects, one class each:

  * describe answers AT ONCE with a ticket, and the status endpoints (by image, by ticket,
    the whole queue) report queued + position, running, done or failed;
  * single-flight: every describe of an image in flight gets the same ticket, and the
    captioner is called once;
  * failure is reported on the ticket -- the captioner's reason, a busy box, an unreadable
    image -- and the old synchronous POST /images/scene keeps its answers (200/503/404);
  * END TO END: a held job is released by the MODAL's describe ticket, with exactly the words
    it saved, whichever came first -- the describe or the job;
  * a hold no longer times out merely for being far back in a long queue;
  * a held job says what it waits for (scene, motion) and where its image is in line, and
    GET /caption-holds sums it up for the JobQueue page.

The captioner is a mock that records, and the line in front of it is held shut by a dataset
caption's turn until a test lets it go -- so "queued, 2nd" is something a test can see.
"""
import asyncio
import inspect
import json
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import caption_hold, caption_tickets
from app import caption_queue as caption_queue_module
from app.auth import get_current_user, verify_api_key, verify_api_key_or_bearer
from app.config import settings
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.joycaption import CaptionError, CaptionerBusy
from app.main import app
from app.models import ImageMeta, Job, Segment, User
from app.routes.captions import ScenePair

IMG = f"s3://{settings.s3_images_bucket}/2026-10-01/00564.png"
OTHER = f"s3://{settings.s3_images_bucket}/2026-10-01/00565.png"
BLOCKER = f"s3://{settings.s3_images_bucket}/2026-10-01/dataset-image.png"
SCENE_WORDS = "a woman in a yellow raincoat standing on a pier, looking at the viewer"
MOTION_WORDS = "she turns toward the sea as the wind lifts her hood, then glances back"
BOTH = "k3lly2026, woman, <SCENE>, <MOTION>"
RECIPE = {"characters": [{"name": "Kelly 564", "trigger": "k3lly2026", "gender": "woman"}]}


# --- fixtures ----------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    q = caption_queue_module.CaptionQueue()
    monkeypatch.setattr(caption_queue_module, "queue", q)
    monkeypatch.setattr(caption_hold, "caption_queue", q)
    monkeypatch.setattr(caption_hold, "JOIN_POLL_S", 0.01)
    monkeypatch.setattr(caption_hold, "BUSY_RETRY_S", 0.01)
    monkeypatch.setattr(settings, "motion_caption_enabled", True)
    caption_hold._waiters.clear()
    caption_hold._notes.clear()
    caption_tickets.reset()
    yield q
    for t in list(caption_hold._waiters.values()):
        t.cancel()
    caption_hold._waiters.clear()
    caption_tickets.reset()


class _Serialized:
    """The test's session, one awaited operation at a time.

    Here the request handlers, the caption tickets and the hold's waiter all run at once, and
    in this file they share one session (so everything rolls back) -- one asyncpg connection,
    which refuses two operations at once. Production gives each its own session.
    """
    def __init__(self, session):
        self._s = session
        self._lock = asyncio.Lock()

    def __getattr__(self, name):
        attr = getattr(self._s, name)
        if not inspect.iscoroutinefunction(attr):
            return attr

        async def locked(*a, **kw):
            async with self._lock:
                return await attr(*a, **kw)
        return locked


@pytest.fixture
def db(db):
    """conftest's rolled-back session, serialized (see _Serialized)."""
    return _Serialized(db)


@pytest_asyncio.fixture
async def shared_session(db, monkeypatch):
    """The tickets' and the hold's short sessions are the test's session: all rolls back.

    Torn down before the session is: whatever a test left in flight is let finish (or
    cancelled), so nothing touches the session after it closes.
    """
    @asynccontextmanager
    async def _session():
        yield db
    monkeypatch.setattr(caption_hold, "async_session", _session)
    monkeypatch.setattr(caption_tickets, "async_session", _session)
    yield db
    pending = [t.task for t in caption_tickets._by_id.values() if t.task and not t.task.done()]
    pending += [w for w in caption_hold._waiters.values() if not w.done()]
    if pending:
        _, still = await asyncio.wait(pending, timeout=2)
        for t in still:
            t.cancel()
        await asyncio.gather(*still, return_exceptions=True)


@pytest.fixture
def captioner(monkeypatch):
    """The captioner, S3 and the settings read, mocked. `.pair` and `.motion` record calls."""
    class C:
        pair = AsyncMock(return_value=ScenePair(
            scene=SCENE_WORDS, scene_instruction="i-scene",
            motion=MOTION_WORDS, motion_instruction="i-motion"))
        motion = AsyncMock(return_value=(MOTION_WORDS, "i-motion"))
        base = AsyncMock(return_value="http://c")

    async def no_settings(db):
        return {}
    monkeypatch.setattr("app.routes.captions.run_caption_pair", C.pair)
    monkeypatch.setattr("app.routes.captions._caption_base", C.base)
    monkeypatch.setattr("app.routes.captions._get_all_settings", no_settings)
    monkeypatch.setattr(caption_tickets, "describe_motion", C.motion)
    monkeypatch.setattr(caption_tickets.s3, "download_bytes", lambda uri: b"png-bytes")
    return C


class Line:
    """The caption queue held shut by a dataset caption at the front, until released."""
    def __init__(self, q):
        self.q = q
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.task = None

    async def __aenter__(self):
        async def hold():
            async with self.q.turn(BLOCKER, kind="dataset"):
                self.entered.set()
                await self.release.wait()
        self.task = asyncio.create_task(hold())
        await self.entered.wait()
        return self

    async def open(self):
        self.release.set()
        await self.task

    async def __aexit__(self, *exc):
        self.release.set()
        await self.task


async def _until(cond, timeout=2.0):
    """Let the background tasks run until `cond()` holds."""
    for _ in range(int(timeout / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for the background tasks")


async def _ticket_queued(path=IMG):
    await _until(lambda: caption_tickets.active(path) is not None)


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user, *, starting_image=IMG, status=JobStatus.PENDING) -> Job:
    job = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=7,
              status=status, starting_image=starting_image)
    db.add(job)
    await db.flush()
    return job


async def _held(db, job, *, prompt=BOTH, status=SegmentStatus.AWAITING_CAPTION) -> Segment:
    seg = Segment(job_id=job.id, index=0, prompt=prompt, status=status, ltx_recipe=RECIPE)
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


@asynccontextmanager
async def _client(db, user=None):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user or object()
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


async def _describe(c, path=IMG, **body):
    resp = await c.post("/images/scene/describe", params={"path": path}, json=body)
    assert resp.status_code == 202, resp.text
    return resp.json()


async def _status(c, path=IMG):
    resp = await c.get("/images/scene/status", params={"path": path})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _segment_row(db, seg_id):
    for obj in list(db.identity_map.values()):
        if isinstance(obj, (Job, Segment)):
            db.expire(obj)
    return await db.get(Segment, seg_id)


async def _claim(db):
    saved = dict(app.dependency_overrides)  # may run inside a _client block
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
        app.dependency_overrides.update(saved)


# --- describe answers at once ------------------------------------------------------------------

@pytest.mark.asyncio
class TestDescribeAnswersAtOnce:
    async def test_it_returns_a_queued_ticket_while_the_line_is_shut(
            self, db, shared_session, captioner, fresh):
        async with Line(fresh) as line, _client(db) as c:
            body = await asyncio.wait_for(_describe(c), 2)
            assert body["status"] == "queued"
            assert body["position"] == 1          # next after the dataset caption
            assert body["depth"] == 2
            assert body["joined"] is False
            assert body["ticket_id"]
            captioner.pair.assert_not_called()

            st = await _status(c)
            assert st["ticket_id"] == body["ticket_id"]
            assert st["status"] == "queued" and st["position"] == 1

            await line.open()
            await caption_tickets.get(body["ticket_id"]).wait()
            done = await _status(c)
            assert done["status"] == "done"
            assert done["position"] is None and done["finished_at"]
            by_id = (await c.get(f"/images/scene/tickets/{body['ticket_id']}")).json()
            assert by_id["status"] == "done"

        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == SCENE_WORDS
        assert meta.motion_description == MOTION_WORDS

    async def test_the_scene_read_carries_the_ticket(self, db, shared_session, captioner,
                                                    fresh):
        async with Line(fresh), _client(db) as c:
            ticket = await _describe(c)
            scene = (await c.get("/images/scene", params={"path": IMG})).json()
            assert scene["caption"]["ticket_id"] == ticket["ticket_id"]
            assert scene["queue_status"] == "queued" and scene["queue_position"] == 1

    async def test_an_unknown_ticket_is_404_and_an_unasked_image_is_null(self, db):
        async with _client(db) as c:
            assert (await c.get("/images/scene/tickets/nope")).status_code == 404
            st = await _status(c, OTHER)
            assert st["status"] is None and st["ticket_id"] is None

    async def test_a_foreign_bucket_is_refused(self, db):
        async with _client(db) as c:
            resp = await c.post("/images/scene/describe", params={"path": "s3://x/y.png"})
            assert resp.status_code == 400


@pytest.mark.asyncio
class TestSingleFlight:
    async def test_two_describes_share_one_ticket_and_one_caption(
            self, db, shared_session, captioner, fresh):
        async with Line(fresh) as line, _client(db) as c:
            a = await _describe(c)
            b = await _describe(c)
            assert b["ticket_id"] == a["ticket_id"]
            assert b["joined"] is True
            assert fresh.depth() == 2  # the dataset caption and ONE describe
            await line.open()
            await caption_tickets.get(a["ticket_id"]).wait()
        assert captioner.pair.await_count == 1

    async def test_after_it_finishes_a_describe_is_a_new_caption(
            self, db, shared_session, captioner):
        async with _client(db) as c:
            a = await _describe(c)
            await caption_tickets.get(a["ticket_id"]).wait()
            b = await _describe(c)
            await caption_tickets.get(b["ticket_id"]).wait()
        assert a["ticket_id"] != b["ticket_id"]
        assert captioner.pair.await_count == 2  # a re-roll is a re-roll

    async def test_the_old_synchronous_post_joins_the_ticket_in_flight(
            self, db, shared_session, captioner, fresh):
        async with Line(fresh) as line, _client(db) as c:
            ticket = await _describe(c)
            sync = asyncio.create_task(c.post("/images/scene", params={"path": IMG}, json={}))
            await asyncio.sleep(0.05)
            assert not sync.done()
            await line.open()
            resp = await asyncio.wait_for(sync, 5)
        assert resp.status_code == 200, resp.text
        assert resp.json()["scene_description"] == SCENE_WORDS
        assert resp.json()["caption"]["ticket_id"] == ticket["ticket_id"]
        assert captioner.pair.await_count == 1


@pytest.mark.asyncio
class TestFailureIsOnTheTicket:
    async def test_a_captioner_error_is_failed_with_the_reason(
            self, db, shared_session, captioner):
        captioner.pair.side_effect = CaptionError("captioner unreachable at http://c")
        async with _client(db) as c:
            t = await _describe(c)
            await caption_tickets.get(t["ticket_id"]).wait()
            st = await _status(c)
        assert st["status"] == "failed"
        assert "unreachable" in st["error"]
        assert st["busy"] is False
        assert (await db.get(ImageMeta, IMG)) is None  # nothing half-written

    async def test_a_busy_box_is_failed_and_says_so(self, db, shared_session, captioner):
        captioner.base.side_effect = CaptionerBusy("3090.zero is in render mode")
        async with _client(db) as c:
            t = await _describe(c)
            await caption_tickets.get(t["ticket_id"]).wait()
            st = await _status(c)
        assert st["status"] == "failed" and st["busy"] is True
        assert "render mode" in st["error"]

    async def test_a_motion_failure_is_done_with_the_scene_kept(
            self, db, shared_session, captioner):
        captioner.pair.return_value = ScenePair(scene=SCENE_WORDS, scene_instruction="i",
                                                motion_error="timed out")
        async with _client(db) as c:
            t = await _describe(c)
            await caption_tickets.get(t["ticket_id"]).wait()
            st = await _status(c)
        assert st["status"] == "done" and st["motion_error"] == "timed out"
        assert (await db.get(ImageMeta, IMG)).scene_description == SCENE_WORDS

    async def test_the_sync_post_keeps_its_status_codes(self, db, shared_session, captioner,
                                                        monkeypatch):
        captioner.pair.side_effect = CaptionError("down")
        async with _client(db) as c:
            assert (await c.post("/images/scene", params={"path": IMG})).status_code == 503

        def gone(uri):
            raise FileNotFoundError(uri)
        monkeypatch.setattr(caption_tickets.s3, "download_bytes", gone)
        caption_tickets.reset()
        async with _client(db) as c:
            assert (await c.post("/images/scene", params={"path": IMG})).status_code == 404


@pytest.mark.asyncio
class TestTheWholeQueueInOnePoll:
    async def test_entries_name_every_place_and_recent_names_the_finished(
            self, db, shared_session, captioner, fresh):
        captioner.pair.side_effect = [
            ScenePair(scene=SCENE_WORDS, scene_instruction="i", motion=MOTION_WORDS),
            CaptionError("boom"),
        ]
        async with Line(fresh) as line, _client(db) as c:
            a = await _describe(c, IMG)
            b = await _describe(c, OTHER)
            q = (await c.get("/images/caption-queue")).json()
            assert [(e["path"], e["kind"], e["status"], e["position"]) for e in q["entries"]] == [
                (BLOCKER, "dataset", "running", 0),
                (IMG, "describe", "queued", 1),
                (OTHER, "describe", "queued", 2),
            ]
            assert q["entries"][1]["ticket_id"] == a["ticket_id"]
            await line.open()
            await caption_tickets.get(a["ticket_id"]).wait()
            await caption_tickets.get(b["ticket_id"]).wait()
            q = (await c.get("/images/caption-queue")).json()
        assert q["entries"] == [] and q["depth"] == 0
        recent = {r["path"]: r for r in q["recent"]}
        assert recent[IMG]["status"] == "done"
        assert recent[OTHER]["status"] == "failed" and recent[OTHER]["error"] == "boom"


# --- the hold and the modal's describe, end to end --------------------------------------------

@pytest.mark.asyncio
class TestAHeldJobIsReleasedByTheModalsDescribe:
    async def test_describe_first_then_the_job(self, db, shared_session, captioner, fresh):
        """The ticket's user flow: Describe in the modal, Use as Starting Image, submit, and
        carry on. The job joins the modal's caption; one caption; released with its words."""
        user = await _user(db)
        async with Line(fresh) as line, _client(db, user) as c:
            ticket = await _describe(c)
            data = {"name": "j", "width": 832, "height": 1216, "fps": 24,
                    "starting_image_uri": IMG,
                    "first_segment": {"prompt": BOTH, "ltx_recipe": RECIPE}}
            resp = await c.post("/jobs", data={"data": json.dumps(data)})
            assert resp.status_code == 201, resp.text
            job_id = uuid.UUID(resp.json()["id"])
            seg = (await db.execute(
                Segment.__table__.select().where(Segment.job_id == job_id))).one()
            assert seg.status == SegmentStatus.AWAITING_CAPTION
            assert await _claim(db) is None

            # Visible while it waits: what it needs, and the image's place in line.
            listed = (await c.get("/jobs")).json()["items"]
            detail = [j for j in listed if j["id"] == str(job_id)][0]["caption_hold_detail"]
            assert detail["needs"] == ["scene", "motion"]
            assert detail["queue_status"] == "queued" and detail["queue_position"] == 1

            await line.open()
            await caption_tickets.get(ticket["ticket_id"]).wait()
            await asyncio.wait_for(caption_hold._waiters[IMG], 5)

        assert captioner.pair.await_count == 1
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.PENDING
        assert row.prompt == f"k3lly2026, woman, {SCENE_WORDS}, {MOTION_WORDS}"
        body = await _claim(db)
        assert body["id"] == str(seg.id)
        assert body["prompt"] == f"k3lly2026, woman, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_job_first_then_the_modal_joins_its_caption(
            self, db, shared_session, captioner, fresh):
        """The other order: the hold's ticket is already in line, and Describe in the modal
        joins it rather than queueing a second caption that would replace the first."""
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh) as line, _client(db) as c:
            waiter = caption_hold.ensure(IMG)
            await _ticket_queued()
            hold_ticket = caption_tickets.active(IMG)
            assert hold_ticket is not None and hold_ticket.origin == "hold"
            body = await _describe(c)
            assert body["ticket_id"] == hold_ticket.id and body["joined"] is True
            await line.open()
            await asyncio.wait_for(waiter, 5)
            st = await _status(c)
        assert st["status"] == "done"
        assert captioner.pair.await_count == 1
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.PENDING
        assert row.prompt == f"k3lly2026, woman, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_a_describe_upgrades_a_queued_motion_only_hold_to_a_reroll(
            self, db, shared_session, captioner, fresh):
        """The hold only needed the paragraph, but a person asked for a fresh description
        before it ran: one caption, the pair, and the job renders the re-rolled words."""
        await _meta(db, scene="the old scene", motion=None)
        captioner.pair.return_value = ScenePair(scene="the re-rolled scene",
                                                scene_instruction="i", motion=MOTION_WORDS)
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh) as line, _client(db) as c:
            waiter = caption_hold.ensure(IMG)
            await _ticket_queued()
            assert caption_tickets.active(IMG).mode == "motion"
            body = await _describe(c)
            assert body["joined"] is True and body["mode"] == "pair"
            await line.open()
            await asyncio.wait_for(waiter, 5)
        captioner.motion.assert_not_called()
        assert captioner.pair.await_count == 1
        row = await _segment_row(db, seg.id)
        assert row.prompt == f"k3lly2026, woman, the re-rolled scene, {MOTION_WORDS}"

    async def test_the_modals_motion_failure_is_made_up_by_the_hold(
            self, db, shared_session, captioner, fresh):
        captioner.pair.return_value = ScenePair(scene=SCENE_WORDS, scene_instruction="i",
                                                motion_error="timed out")
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh) as line, _client(db) as c:
            await _describe(c)
            waiter = caption_hold.ensure(IMG)
            await line.open()
            await asyncio.wait_for(waiter, 5)
        captioner.motion.assert_awaited_once()
        assert captioner.motion.await_args.args[1] == SCENE_WORDS  # grounded on the saved scene
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.PENDING
        assert row.prompt == f"k3lly2026, woman, {SCENE_WORDS}, {MOTION_WORDS}"

    async def test_the_modals_failed_caption_is_retried_by_the_hold(
            self, db, shared_session, captioner, fresh):
        captioner.pair.side_effect = [
            CaptionError("a hiccup"),
            ScenePair(scene=SCENE_WORDS, scene_instruction="i", motion=MOTION_WORDS),
        ]
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh) as line, _client(db) as c:
            await _describe(c)
            waiter = caption_hold.ensure(IMG)
            await line.open()
            await asyncio.wait_for(waiter, 5)
        assert captioner.pair.await_count == 2
        assert (await _segment_row(db, seg.id)).status == SegmentStatus.PENDING


@pytest.mark.asyncio
class TestALongQueueIsNotATimeout:
    async def test_joining_a_describe_far_back_in_line_does_not_fail_the_hold(
            self, db, shared_session, captioner, fresh, monkeypatch):
        """console#562's first version timed a join out after caption_hold_timeout_s (1 h).
        At ~6 min an image and thirty in line, that failed jobs whose caption was simply
        on its way. The limit is for a captioner that keeps refusing, not for a queue."""
        monkeypatch.setattr(settings, "caption_hold_timeout_s", 0)
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh) as line, _client(db) as c:
            await _describe(c)
            waiter = caption_hold.ensure(IMG)
            await asyncio.sleep(0.1)
            assert (await _segment_row(db, seg.id)).status == SegmentStatus.AWAITING_CAPTION
            await line.open()
            await asyncio.wait_for(waiter, 5)
        assert (await _segment_row(db, seg.id)).status == SegmentStatus.PENDING

    async def test_a_captioner_that_keeps_refusing_still_fails_it(
            self, db, shared_session, captioner, monkeypatch):
        monkeypatch.setattr(settings, "caption_hold_timeout_s", 0)
        captioner.base.side_effect = CaptionerBusy("3090.zero is in render mode")
        seg = await _held(db, await _job(db, await _user(db)))
        await asyncio.wait_for(caption_hold.ensure(IMG), 5)
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.CAPTION_FAILED
        assert "render mode" in row.error_message


# --- what a held job shows --------------------------------------------------------------------

@pytest.mark.asyncio
class TestWhatAHeldJobShows:
    async def test_the_detail_says_which_half_and_where_in_line(
            self, db, shared_session, captioner, fresh):
        user = await _user(db)
        await _meta(db, motion=None)                   # scene saved, motion missing
        job = await _job(db, user)
        await _held(db, job)
        async with Line(fresh), _client(db, user) as c:
            await _describe(c, OTHER)                  # someone else is ahead
            caption_hold.ensure(IMG)
            await _ticket_queued()
            detail = (await c.get(f"/jobs/{job.id}")).json()
        seg = detail["segments"][0]
        assert seg["caption_needs"] == ["motion"]
        assert seg["caption_queue_status"] == "queued"
        assert seg["caption_queue_position"] == 2
        assert seg["caption_queue_depth"] == 3
        assert "#2 of 3" in seg["caption_wait"]
        assert detail["caption_hold_detail"]["needs"] == ["motion"]
        assert detail["caption_hold_detail"]["queue_position"] == 2

    async def test_a_hold_between_refusals_says_waiting_and_why(
            self, db, shared_session, captioner, monkeypatch):
        monkeypatch.setattr(caption_hold, "BUSY_RETRY_S", 30)
        captioner.base.side_effect = CaptionerBusy("3090.zero is in render mode")
        user = await _user(db)
        job = await _job(db, user)
        await _held(db, job)
        caption_hold.ensure(IMG)
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "unavailable" in (caption_hold._notes.get(IMG) or ""):
                break
        async with _client(db, user) as c:
            detail = (await c.get(f"/jobs/{job.id}")).json()
        seg = detail["segments"][0]
        assert seg["caption_queue_status"] == "waiting"
        assert "render mode" in seg["caption_wait"]

    async def test_the_summary_counts_jobs_and_names_the_queue(
            self, db, shared_session, captioner, fresh):
        user = await _user(db)
        a = await _job(db, user)
        await _held(db, a)
        b = await _job(db, user)
        await _held(db, b)
        c_job = await _job(db, user, starting_image=OTHER)
        await _held(db, c_job, status=SegmentStatus.CAPTION_FAILED)
        async with Line(fresh), _client(db, user) as c:
            caption_hold.ensure(IMG)
            await _ticket_queued()
            s = (await c.get("/caption-holds")).json()
        assert s["jobs_waiting"] >= 2 and s["segments_waiting"] >= 2
        assert s["jobs_failed"] >= 1
        assert s["queue_depth"] == 2 and s["running"] == BLOCKER
        mine = [i for i in s["images"] if i["image"] == IMG][0]
        assert mine["needs"] == ["scene", "motion"]
        assert mine["jobs"] == 2 and mine["segments"] == 2
        assert mine["queue_status"] == "queued" and mine["queue_position"] == 1


# --- release does not wait for the front of the line ------------------------------------------

@pytest.mark.asyncio
class TestTheSweepReleasesWhatIsAlreadySaved:
    """Seen live on 2026-10-02: a deploy's startup sweep released two held segments at once,
    words long since saved. Their waiters had been sitting in a ~30-deep caption queue, and
    a waiter only looked at the row again at the front of the line -- so a segment whose
    words were saved some other way stayed held for as long as the line was long."""

    async def test_words_saved_while_the_hold_waits_in_line_release_on_the_next_sweep(
            self, db, shared_session, captioner, fresh):
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh):
            waiter = caption_hold.ensure(IMG)
            await _ticket_queued()
            hold_ticket = caption_tickets.active(IMG)
            # Saved behind the hold's back: the bulk-tag auto-describe writes the row directly.
            await _meta(db)
            assert (await _segment_row(db, seg.id)).status == SegmentStatus.AWAITING_CAPTION

            await caption_hold.sweep()                 # the line is still shut

            row = await _segment_row(db, seg.id)
            assert row.status == SegmentStatus.PENDING
            assert row.prompt == f"k3lly2026, woman, {SCENE_WORDS}, {MOTION_WORDS}"
            # And the hold's place in line is given back rather than captioned for nothing.
            assert hold_ticket.withdrawn and caption_tickets.active(IMG) is None
            assert fresh.depth() == 1
            await asyncio.wait_for(waiter, 2)
        captioner.pair.assert_not_called()
        # A withdrawn ticket is not a failure the image should show.
        assert caption_tickets.latest(IMG) is None

    async def test_a_reroll_in_flight_is_still_waited_for(
            self, db, shared_session, captioner, fresh):
        """gate()'s rule holds in the sweep too: a person's re-roll is about to replace the
        saved words, and the job renders the words on the screen."""
        await _meta(db, scene="the old scene", motion="the old motion")
        captioner.pair.return_value = ScenePair(scene="the new scene", scene_instruction="i",
                                                motion="the new motion")
        seg = await _held(db, await _job(db, await _user(db)))
        async with Line(fresh) as line, _client(db) as c:
            await _describe(c)
            waiter = caption_hold.ensure(IMG)
            await caption_hold.sweep()
            assert (await _segment_row(db, seg.id)).status == SegmentStatus.AWAITING_CAPTION
            await line.open()
            await asyncio.wait_for(waiter, 5)
        row = await _segment_row(db, seg.id)
        assert row.prompt == "k3lly2026, woman, the new scene, the new motion"
