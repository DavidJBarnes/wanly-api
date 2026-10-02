"""Captions per half (console#590).

Tagging makes the scene only; "Describe motion" makes the motion only; each half can be redone
or retried on its own and saving one never clears or overwrites the other. Motion is grounded
on the SAVED scene. A held job asks for exactly the half its prompt needs, and the ticket says
which job asked ("Motion requested by job ...").

Fixtures (the mocked captioner, the shut Line, the shared session) are test_caption_tickets'.
"""
import asyncio

import pytest

from app import caption_hold, caption_tickets
from app.config import settings
from app.enums import SegmentStatus
from app.joycaption import CaptionError
from app.models import ImageMeta
from app.routes.captions import ScenePair
from tests.test_caption_tickets import (  # noqa: F401 - fixtures are used by name
    IMG, MOTION_WORDS, SCENE_WORDS, Line, _all_done, _client, _describe, _held, _job, _meta,
    _segment_row, _status, _ticket_queued, _user, captioner, db, fresh,
    shared_session,
)

MOTION_ONLY = "k3lly2026, woman, <MOTION>"
SCENE_ONLY = "k3lly2026, woman, <SCENE>"


@pytest.mark.asyncio
class TestEachHalfOnItsOwn:
    async def test_scene_only_makes_the_scene_and_keeps_the_saved_motion(
            self, db, shared_session, captioner):
        await _meta(db, scene="the old scene", motion="the old motion")
        async with _client(db) as c:
            body = await _describe(c, halves=["scene"])
            assert [t["half"] for t in body["tickets"]] == ["scene"]
            assert caption_tickets.active(IMG, "motion") is None
            await _all_done(body)
        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == SCENE_WORDS
        assert meta.motion_description == "the old motion"  # never cleared
        captioner.motion.assert_not_called()

    async def test_motion_only_is_grounded_on_the_saved_scene_and_keeps_it(
            self, db, shared_session, captioner):
        await _meta(db, scene="the saved scene", motion=None)
        async with _client(db) as c:
            body = await _describe(c, halves=["motion"])
            assert body["half"] == "motion" and body["lane"] == "motion"
            await _all_done(body)
        captioner.pair.assert_not_called()               # no scene was made
        assert captioner.motion.await_args.args[1] == "the saved scene"
        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == "the saved scene"
        assert meta.motion_description == MOTION_WORDS

    async def test_motion_with_no_scene_asks_for_the_scene_first(
            self, db, shared_session, captioner):
        async with _client(db) as c:
            body = await _describe(c, halves=["motion"])
            await _all_done(body)
        assert captioner.pair.await_count == 1
        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == SCENE_WORDS
        assert meta.motion_description == MOTION_WORDS

    async def test_motion_asked_during_a_scene_redo_grounds_on_the_new_scene(
            self, db, shared_session, captioner, fresh, monkeypatch):
        await _meta(db, scene="the old scene", motion=None)
        grounded = []

        async def motion(image, scene_text, **kw):
            grounded.append(scene_text)
            return MOTION_WORDS, "i"
        monkeypatch.setattr(caption_tickets, "describe_motion", motion)
        async with Line(fresh) as line, _client(db) as c:
            scene = await _describe(c, halves=["scene"])
            mot = await _describe(c, halves=["motion"])
            await line.open()
            await _all_done(scene)
            await _all_done(mot)
        assert grounded == [SCENE_WORDS]

    async def test_each_half_is_single_flight_and_the_halves_are_separate(
            self, db, shared_session, captioner, fresh):
        await _meta(db, motion=None)
        async with Line(fresh) as line, _client(db) as c:
            m1 = await _describe(c, halves=["motion"])
            m2 = await _describe(c, halves=["motion"])
            s1 = await _describe(c, halves=["scene"])
            assert m2["ticket_id"] == m1["ticket_id"] and m2["joined"] is True
            assert s1["ticket_id"] != m1["ticket_id"] and s1["joined"] is False
            st = await _status(c)
            assert st["scene"]["ticket_id"] == s1["ticket_id"]
            assert st["motion"]["ticket_id"] == m1["ticket_id"]
            await line.open()
            await _all_done(m1)
            await _all_done(s1)

    async def test_a_failed_motion_is_retried_alone(self, db, shared_session, captioner):
        await _meta(db, scene="the saved scene", motion=None)
        captioner.motion.side_effect = [CaptionError("timed out"), (MOTION_WORDS, "i")]
        async with _client(db) as c:
            first = await _describe(c, halves=["motion"])
            await _all_done(first)
            failed = await _status(c, half="motion")
            assert failed["status"] == "failed" and failed["error"] == "timed out"
            assert (await _status(c, half="scene"))["status"] is None
            retry = await _describe(c, halves=["motion"])
            assert retry["ticket_id"] != first["ticket_id"]
            await _all_done(retry)
            assert (await _status(c, half="motion"))["status"] == "done"
        captioner.pair.assert_not_called()
        meta = await db.get(ImageMeta, IMG)
        assert meta.scene_description == "the saved scene"
        assert meta.motion_description == MOTION_WORDS

    async def test_a_scene_redo_leaves_the_motion_older_than_the_scene(
            self, db, shared_session, captioner):
        """What the console's "Redo motion to re-ground on the new scene" hint reads."""
        async with _client(db) as c:
            await _all_done(await _describe(c))
            before = await db.get(ImageMeta, IMG)
            motion_at = before.motion_described_at
            await _all_done(await _describe(c, halves=["scene"]))
            scene = (await c.get("/images/scene", params={"path": IMG})).json()
        assert scene["motion_description"] == MOTION_WORDS
        assert scene["motion_described_at"] is not None
        meta = await db.get(ImageMeta, IMG)
        assert meta.motion_described_at == motion_at
        assert meta.scene_described_at > meta.motion_described_at
        assert scene["scene_caption"]["half"] == "scene"
        assert scene["motion_caption"]["half"] == "motion"

    async def test_an_unknown_half_is_refused(self, db):
        async with _client(db) as c:
            resp = await c.post("/images/scene/describe", params={"path": IMG},
                                json={"halves": ["both"]})
        assert resp.status_code == 422

    async def test_motion_switched_off_fails_the_motion_ticket_and_says_why(
            self, db, shared_session, captioner, monkeypatch):
        monkeypatch.setattr(settings, "motion_caption_enabled", False)
        await _meta(db, motion=None)
        async with _client(db) as c:
            body = await _describe(c, halves=["motion"])
            await _all_done(body)
            st = await _status(c, half="motion")
        assert st["status"] == "failed" and "switched off" in st["error"]


@pytest.mark.asyncio
class TestHeldJobsAskForTheHalfTheyNeed:
    async def test_a_motion_recipe_job_asks_for_the_motion_alone_and_says_so(
            self, db, shared_session, captioner, fresh):
        await _meta(db, scene="the saved scene", motion=None)
        user = await _user(db)
        job = await _job(db, user)
        seg = await _held(db, job, prompt=MOTION_ONLY)
        async with Line(fresh) as line, _client(db, user) as c:
            waiter = caption_hold.ensure(IMG)
            await _ticket_queued()
            assert caption_tickets.active(IMG, "scene") is None
            t = caption_tickets.active(IMG, "motion")
            assert t is not None and t.origin == "hold"
            st = await _status(c, half="motion")
            assert st["requested_by"] == [{"job_id": str(job.id), "name": job.name}]
            q = (await c.get("/images/caption-queue")).json()
            mine = [e for e in q["entries"] if e["path"] == IMG]
            assert [(e["kind"], e["lane"]) for e in mine] == [("hold", "motion")]
            assert mine[0]["requested_by"] == [{"job_id": str(job.id), "name": job.name}]
            await line.open()
            await asyncio.wait_for(waiter, 5)
        captioner.pair.assert_not_called()
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.PENDING
        assert row.prompt == f"k3lly2026, woman, {MOTION_WORDS}"

    async def test_a_scene_only_job_never_asks_for_a_motion(
            self, db, shared_session, captioner, fresh):
        seg = await _held(db, await _job(db, await _user(db)), prompt=SCENE_ONLY)
        async with Line(fresh) as line:
            waiter = caption_hold.ensure(IMG)
            await _ticket_queued()
            assert caption_tickets.active(IMG, "scene").origin == "hold"
            assert caption_tickets.active(IMG, "motion") is None
            await line.open()
            await asyncio.wait_for(waiter, 5)
        captioner.motion.assert_not_called()
        meta = await db.get(ImageMeta, IMG)
        assert meta.motion_description is None
        assert (await _segment_row(db, seg.id)).status == SegmentStatus.PENDING

    async def test_a_motion_redo_in_flight_holds_a_job_that_uses_the_motion(
            self, db, shared_session, captioner, fresh):
        """gate()'s rule per half: the redone paragraph is the one on the person's screen."""
        await _meta(db, scene="the saved scene", motion="the old motion")
        captioner.motion.return_value = ("the new motion", "i")
        seg = await _held(db, await _job(db, await _user(db)), prompt=MOTION_ONLY)
        async with Line(fresh) as line, _client(db) as c:
            body = await _describe(c, halves=["motion"])
            waiter = caption_hold.ensure(IMG)
            await caption_hold.sweep()
            assert (await _segment_row(db, seg.id)).status == SegmentStatus.AWAITING_CAPTION
            await line.open()
            await _all_done(body)
            await asyncio.wait_for(waiter, 5)
        row = await _segment_row(db, seg.id)
        assert row.prompt == "k3lly2026, woman, the new motion"

    async def test_a_motion_redo_does_not_hold_a_job_that_uses_only_the_scene(
            self, db, shared_session, captioner, fresh):
        await _meta(db, scene="the saved scene", motion="the old motion")
        seg = await _held(db, await _job(db, await _user(db)), prompt=SCENE_ONLY)
        async with Line(fresh), _client(db) as c:
            await _describe(c, halves=["motion"])
            await asyncio.wait_for(caption_hold.ensure(IMG), 5)
            row = await _segment_row(db, seg.id)
            assert row.status == SegmentStatus.PENDING
            assert row.prompt == "k3lly2026, woman, the saved scene"
