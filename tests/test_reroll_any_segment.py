"""Re-rolling a continuation, and re-rolling with a different prompt (console#424).

Re-roll served index 0 only, and changed nothing but the seed. Both limits are lifted here,
and the things that can go wrong are database things — which take gets archived, which index
the replacement lands on, and what happens to the seeds of takes NOT being rolled. That last
one is the subtle one: a live segment holds seed NULL, meaning "ask the job", and rolling a
continuation moves the job's seed out from under it.
"""
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.auth import get_current_user
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User, Wildcard


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _chain(db, user, *, length=3, status=JobStatus.AWAITING,
                 recipe=None) -> tuple[Job, list[Segment]]:
    """A job of `length` completed segments — the shape re-roll could not touch before."""
    job = Job(user_id=user.id, name="job", width=480, height=832, fps=16, seed=1234,
              starting_image="s3://b/start.png", status=status)
    db.add(job)
    await db.flush()
    segments = []
    for i in range(length):
        seg = Segment(
            job_id=job.id, index=i, prompt=f"take {i}",
            duration_seconds=5.0, speed=1.0, status=SegmentStatus.COMPLETED,
            output_path=f"s3://b/{i}.mp4", last_frame_path=f"s3://b/{i}.png",
            ltx_recipe=recipe if recipe is not None else {"recipe": "Missionary Side"},
        )
        db.add(seg)
        segments.append(seg)
    await db.flush()
    return job, segments


async def _post(db, user, url, json=None):
    for obj in list(db.identity_map.values()):
        if isinstance(obj, (Job, Segment)):
            db.expire(obj)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            return await c.post(url, json=json)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestRollingAContinuation:
    async def test_the_last_segment_can_be_rolled(self, db):
        user = await _user(db)
        job, segs = await _chain(db, user)
        last_id = segs[-1].id

        resp = await _post(db, user, f"/segments/{last_id}/reroll")

        assert resp.status_code == 200, resp.text
        assert resp.json()["index"] == 2
        rows = (await db.execute(
            select(Segment).where(Segment.job_id == job.id, Segment.index == 2)
        )).scalars().all()
        assert {r.discarded for r in rows} == {True, False}
        assert next(r for r in rows if not r.discarded).status == SegmentStatus.PENDING

    async def test_an_earlier_segment_is_refused(self, db):
        """The protection that used to read "index must be 0", stated generally: segment 2
        continues from segment 1's last frame, so replacing 1 would orphan it."""
        user = await _user(db)
        _, segs = await _chain(db, user)

        resp = await _post(db, user, f"/segments/{segs[1].id}/reroll")

        assert resp.status_code == 400
        assert "current segment" in resp.json()["detail"]

    async def test_the_predecessors_keep_the_seed_they_ran_on(self, db):
        """The subtle one. A live take holds seed NULL, which means "ask the job", and this
        roll gives the job a new seed. Without stamping them first, retrying segment 0 would
        re-claim it, be handed the NEW seed, and render a different take from the one it is
        retrying — silently."""
        user = await _user(db)
        job, segs = await _chain(db, user)
        original = job.seed

        await _post(db, user, f"/segments/{segs[-1].id}/reroll")

        for seg in segs[:-1]:
            await db.refresh(seg)
            assert seg.seed == original
        await db.refresh(job)
        assert job.seed != original

    async def test_the_archived_take_keeps_its_own_seed(self, db):
        user = await _user(db)
        job, segs = await _chain(db, user)
        original = job.seed

        await _post(db, user, f"/segments/{segs[-1].id}/reroll")

        await db.refresh(segs[-1])
        assert segs[-1].discarded is True
        assert segs[-1].seed == original

    async def test_a_finalized_job_is_refused(self, db):
        """Its stitched output describes the takes that were live when it was built."""
        user = await _user(db)
        _, segs = await _chain(db, user, status=JobStatus.FINALIZED)

        resp = await _post(db, user, f"/segments/{segs[-1].id}/reroll")

        assert resp.status_code == 400
        assert "cannot be re-rolled" in resp.json()["detail"]

    async def test_a_running_segment_is_refused(self, db):
        user = await _user(db)
        job, segs = await _chain(db, user)
        segs[-1].status = SegmentStatus.PROCESSING
        await db.flush()

        resp = await _post(db, user, f"/segments/{segs[-1].id}/reroll")

        assert resp.status_code == 400
        assert "Cancel it" in resp.json()["detail"]

    async def test_another_users_segment_is_not_found(self, db):
        owner = await _user(db)
        intruder = await _user(db)
        _, segs = await _chain(db, owner)

        assert (await _post(db, intruder, f"/segments/{segs[-1].id}/reroll")).status_code == 404


@pytest.mark.asyncio
class TestRollingWithANewPrompt:
    async def test_the_new_take_carries_the_new_prompt(self, db):
        user = await _user(db)
        _, segs = await _chain(db, user)

        body = (await _post(db, user, f"/segments/{segs[-1].id}/reroll",
                            {"prompt": "she turns to face him"})).json()

        assert body["prompt"] == "she turns to face him"

    async def test_the_archived_take_keeps_the_prompt_it_ran(self, db):
        """Otherwise the archive stops answering the only question it exists for."""
        user = await _user(db)
        _, segs = await _chain(db, user)

        await _post(db, user, f"/segments/{segs[-1].id}/reroll", {"prompt": "something else"})

        await db.refresh(segs[-1])
        assert segs[-1].prompt == "take 2"

    async def test_no_prompt_means_a_seed_only_roll(self, db):
        """The default, and what re-roll has always been: one variable, two comparable takes."""
        user = await _user(db)
        _, segs = await _chain(db, user)

        body = (await _post(db, user, f"/segments/{segs[-1].id}/reroll")).json()

        assert body["prompt"] == "take 2"
        assert (body["ltx_recipe"] or {}).get("edited") is None

    async def test_a_changed_prompt_is_recorded_on_the_recipe(self, db):
        """Six takes later, a prompt-changed pair is indistinguishable from a seed-only one,
        and a judgement made across them is worthless."""
        user = await _user(db)
        _, segs = await _chain(db, user)

        body = (await _post(db, user, f"/segments/{segs[-1].id}/reroll",
                            {"prompt": "different words"})).json()

        assert "prompt" in body["ltx_recipe"]["edited"]

    async def test_marking_the_new_take_does_not_rewrite_the_archived_one(self, db):
        """They share the recipe dict; mutating it would rewrite the record of a take that
        was never edited."""
        user = await _user(db)
        _, segs = await _chain(db, user, recipe={"recipe": "Missionary Side", "edited": []})

        await _post(db, user, f"/segments/{segs[-1].id}/reroll", {"prompt": "different"})

        await db.refresh(segs[-1])
        assert segs[-1].ltx_recipe["edited"] == []

    async def test_wildcards_in_a_new_prompt_are_resolved(self, db):
        """A prompt typed here must behave exactly like the same words typed into the create
        dialog. Stored raw, <face> would reach the model as literal text."""
        user = await _user(db)
        name = f"face{uuid.uuid4().hex[:6]}"
        db.add(Wildcard(name=name, options=["smiling"]))
        _, segs = await _chain(db, user)
        await db.flush()

        body = (await _post(db, user, f"/segments/{segs[-1].id}/reroll",
                            {"prompt": f"she is <{name}>"})).json()

        assert body["prompt"] == "she is smiling"
        assert body["prompt_template"] == f"she is <{name}>"
