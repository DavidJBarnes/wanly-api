"""Caption halves after the motion half was removed (wanly-api#481).

There used to be a scene AND a motion half (console#590), each its own ticket. Only the scene
is left. What this protects is how an OLDER console that still asks for "motion" is answered:
beside "scene" it is dropped, on its own it is a 422 that says why, and a status read with
?half=motion is an ordinary "nothing asked for" rather than an error. Held jobs whose prompts
still carry a legacy <MOTION> are released without it, never held for it.

Fixtures (the mocked captioner, the shut Line, the shared session) are test_caption_tickets'.
"""
import asyncio

import pytest

from app import caption_hold, caption_tickets
from app.enums import SegmentStatus
from app.models import ImageMeta
from tests.test_caption_tickets import (  # noqa: F401 - fixtures are used by name
    IMG, SCENE_WORDS, Line, _all_done, _client, _describe, _held, _job, _meta,
    _segment_row, _status, _user, captioner, db, fresh, shared_session,
)


@pytest.mark.asyncio
class TestAnOlderConsoleAskingForMotion:
    async def test_scene_and_motion_makes_the_scene_only(self, db, shared_session, captioner):
        async with _client(db) as c:
            body = await _describe(c, halves=["scene", "motion"])
            assert body["half"] == "scene"
            assert [t["half"] for t in body["tickets"]] == ["scene"]
            await _all_done(body)
        assert (await db.get(ImageMeta, IMG)).scene_description == SCENE_WORDS
        assert captioner.pair.await_count == 1

    async def test_motion_alone_is_refused_and_says_why(self, db, shared_session, captioner):
        async with _client(db) as c:
            resp = await c.post("/images/scene/describe", params={"path": IMG},
                                json={"halves": ["motion"]})
            sync = await c.post("/images/scene", params={"path": IMG},
                                json={"halves": ["motion"]})
        assert resp.status_code == 422 and "motion" in resp.json()["detail"].lower()
        assert sync.status_code == 422
        assert caption_tickets.active(IMG) is None
        captioner.pair.assert_not_called()

    async def test_an_unknown_half_is_refused(self, db):
        async with _client(db) as c:
            resp = await c.post("/images/scene/describe", params={"path": IMG},
                                json={"halves": ["both"]})
        assert resp.status_code == 422

    async def test_status_for_the_motion_half_is_null_not_an_error(
            self, db, shared_session, captioner):
        async with _client(db) as c:
            await _all_done(await _describe(c))
            motion = await _status(c, half="motion")
            scene = await _status(c, half="scene")
        assert motion["status"] is None and motion["ticket_id"] is None
        assert motion["scene"]["status"] == "done"      # the scene's ticket still carried
        assert "motion" not in motion
        assert scene["status"] == "done" and scene["half"] == "scene"

    async def test_the_scene_read_has_no_motion_fields(self, db, shared_session, captioner):
        await _meta(db)
        async with _client(db) as c:
            scene = (await c.get("/images/scene", params={"path": IMG})).json()
        assert scene["scene_description"] == SCENE_WORDS
        assert not [k for k in scene if k.startswith("motion")]


@pytest.mark.asyncio
class TestLegacyMotionInAHeldJob:
    async def test_a_scene_and_motion_job_is_released_with_the_scene_and_no_motion(
            self, db, shared_session, captioner, fresh):
        seg = await _held(db, await _job(db, await _user(db)),
                          prompt="k3lly2026, woman, <SCENE>, <MOTION>")
        async with Line(fresh) as line:
            waiter = caption_hold.ensure(IMG)
            await line.open()
            await asyncio.wait_for(waiter, 5)
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.PENDING
        assert row.prompt == f"k3lly2026, woman, {SCENE_WORDS}"
        assert captioner.pair.await_count == 1

    async def test_a_motion_only_job_is_released_by_the_sweep_without_a_caption(
            self, db, shared_session, captioner):
        seg = await _held(db, await _job(db, await _user(db)),
                          prompt="k3lly2026, woman, she smiles at the camera, <MOTION>")
        await caption_hold.sweep()
        row = await _segment_row(db, seg.id)
        assert row.status == SegmentStatus.PENDING
        assert row.prompt == "k3lly2026, woman, she smiles at the camera"
        captioner.pair.assert_not_called()
        assert caption_tickets.active(IMG) is None
