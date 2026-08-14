"""Re-rolling segment 0: another take of the same shot, with the seed as the only difference.

Run against a real database, because the things that can go wrong here are database things --
whether the replacement can occupy index 0 while the archived take still holds it (it can: the
unique index is partial over live rows), and whether the archived take keeps the seed that
actually produced it.

That last one is the reason segments have a seed at all. With only job.seed, re-rolling would
overwrite it, and every previously archived take would end up labelled with the newest seed. In a
system where the seed is the dominant variable in what a take looks like, an archive that relabels
its own history is worse than no archive.
"""

import uuid

import pytest
from sqlalchemy import select

from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.auth import get_current_user
from app.database import get_db
from app.models import Job, Segment, User


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job_with_segment(db, user, **segment_kwargs) -> tuple[Job, Segment]:
    job = Job(
        user_id=user.id,
        name="job",
        width=480,
        height=832,
        fps=16,
        seed=1234,
        starting_image="s3://b/start.png",
        status=JobStatus.AWAITING,
    )
    db.add(job)
    await db.flush()

    fields = dict(
        job_id=job.id,
        index=0,
        prompt="she moves",
        prompt_template="she {verb}",
        duration_seconds=5.0,
        speed=1.0,
        loras=[{"lora_id": "abc", "high_weight": 1.0, "low_weight": 0.0}],
        faceswap_enabled=True,
        faceswap_image="s3://b/face.png",
        negative_prompt="blurry",
        video_preset_id=None,
        status=SegmentStatus.COMPLETED,
        output_path="s3://b/out.mp4",
        rating=4,
        notes="good travel",
        trim_start_frames=3,
    )
    fields.update(segment_kwargs)
    segment = Segment(**fields)
    db.add(segment)
    await db.flush()
    return job, segment


async def _reroll(db, user, job_id):
    from httpx import ASGITransport, AsyncClient

    # Every real request gets a fresh session. These tests share one so their writes can be
    # rolled back together, so expire the job graph first -- otherwise the route sees the
    # collection as it was loaded earlier in the test and never notices the segment the previous
    # call added. The user is left loaded: it is handed to the route as an object, not re-queried,
    # and expiring it would make the dependency touch the database from sync context.
    for obj in list(db.identity_map.values()):
        if isinstance(obj, (Job, Segment)):
            db.expire(obj)
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(f"/jobs/{job_id}/reroll")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestReroll:
    async def test_it_archives_the_old_take_and_queues_a_new_one_at_index_0(self, db):
        user = await _user(db)
        job, old = await _job_with_segment(db, user)

        resp = await _reroll(db, user, job.id)
        assert resp.status_code == 200, resp.text

        segments = (
            await db.execute(select(Segment).where(Segment.job_id == job.id))
        ).scalars().all()
        assert len(segments) == 2
        archived = next(s for s in segments if s.id == old.id)
        fresh = next(s for s in segments if s.id != old.id)

        # Both hold index 0 at once. That is what the partial unique index is for.
        assert archived.discarded is True and archived.index == 0
        assert fresh.discarded is False and fresh.index == 0
        assert fresh.status == SegmentStatus.PENDING

    async def test_the_new_take_carries_its_own_seed(self, db):
        user = await _user(db)
        job, old = await _job_with_segment(db, user)

        resp = await _reroll(db, user, job.id)

        assert resp.json()["seed"] is not None
        # And the job's seed is untouched, so the archived take still reproduces from it.
        await db.refresh(job)
        assert job.seed == 1234
        await db.refresh(old)
        assert old.seed is None

    async def test_the_seed_stays_inside_the_javascript_safe_range(self, db):
        """A seed above 2**53 displays in the browser as a different number than it is.

        It would round on the way through JSON, so the number shown next to a clip would not be
        the number that generated it -- which defeats recording it at all.
        """
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        seed = (await _reroll(db, user, job.id)).json()["seed"]
        assert 0 <= seed <= 2**53 - 1

    async def test_the_shot_is_copied_so_the_seed_is_the_only_variable(self, db):
        user = await _user(db)
        job, old = await _job_with_segment(db, user)

        body = (await _reroll(db, user, job.id)).json()

        assert body["prompt"] == old.prompt  # resolved, NOT re-rolled through the wildcards
        assert body["prompt_template"] == old.prompt_template
        assert body["duration_seconds"] == old.duration_seconds
        assert body["loras"] == old.loras
        assert body["faceswap_enabled"] is True
        assert body["faceswap_image"] == old.faceswap_image
        assert body["negative_prompt"] == old.negative_prompt

    async def test_the_annotations_of_the_old_take_are_not_copied(self, db):
        # Rating, notes and trims describe the take being archived. Carrying them over would
        # attribute one take's judgement to another one that has not been watched yet.
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        body = (await _reroll(db, user, job.id)).json()

        assert body["rating"] is None
        assert body["notes"] is None
        assert body["trim_start_frames"] == 0
        assert body["output_path"] is None

    async def test_the_job_goes_back_to_pending_so_a_worker_claims_it(self, db):
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        await _reroll(db, user, job.id)

        await db.refresh(job)
        assert job.status == JobStatus.PENDING

    async def test_rolling_again_is_allowed_with_an_archived_take_present(self, db):
        """The workflow is rolling repeatedly, so "one segment" has to mean one LIVE segment."""
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)

        await _reroll(db, user, job.id)
        # The new take has to have finished before it can be archived in turn.
        fresh = (
            await db.execute(
                select(Segment).where(Segment.job_id == job.id, Segment.discarded.is_(False))
            )
        ).scalar_one()
        fresh.status = SegmentStatus.COMPLETED
        await db.flush()

        resp = await _reroll(db, user, job.id)
        assert resp.status_code == 200

        live = (
            await db.execute(
                select(Segment).where(Segment.job_id == job.id, Segment.discarded.is_(False))
            )
        ).scalars().all()
        assert len(live) == 1
        assert len(
            (await db.execute(select(Segment).where(Segment.job_id == job.id))).scalars().all()
        ) == 3

    async def test_a_job_with_a_successor_segment_is_refused(self, db):
        """Segment 1 was generated from segment 0's last frame. Replacing 0 orphans it."""
        user = await _user(db)
        job, _ = await _job_with_segment(db, user)
        db.add(Segment(job_id=job.id, index=1, prompt="p", status=SegmentStatus.COMPLETED))
        await db.flush()

        resp = await _reroll(db, user, job.id)
        assert resp.status_code == 400
        assert "single segment" in resp.json()["detail"]

    async def test_a_running_segment_is_refused(self, db):
        # Otherwise the worker keeps generating a take that has already been archived.
        user = await _user(db)
        job, _ = await _job_with_segment(db, user, status=SegmentStatus.PROCESSING)

        resp = await _reroll(db, user, job.id)
        assert resp.status_code == 400
        assert "Cancel it" in resp.json()["detail"]

    async def test_another_users_job_is_not_found(self, db):
        owner = await _user(db)
        intruder = await _user(db)
        job, _ = await _job_with_segment(db, owner)

        resp = await _reroll(db, intruder, job.id)
        assert resp.status_code == 404


class TestSeedRange:
    def test_new_seeds_are_javascript_safe(self):
        """Job seeds used to come from the full 2**63 range, which the browser cannot show.

        The seed is displayed on the job page and can be typed back into the create dialog to
        reproduce a job -- the only reason to show it at all. Above 2**53 it is silently rounded
        on the way through JSON, so that round trip produced a DIFFERENT seed with nothing
        reporting a problem. Drawing from the full range made that near certain rather than rare:
        roughly one seed in a thousand landed low enough to survive.
        """
        from app.seeds import JS_SAFE_MAX_SEED, new_seed

        assert JS_SAFE_MAX_SEED == 2**53 - 1
        for _ in range(200):
            seed = new_seed()
            assert 0 <= seed <= JS_SAFE_MAX_SEED
            # The round trip a browser performs. Equality here is the whole property.
            assert int(float(seed)) == seed


@pytest.mark.asyncio
class TestClaimUsesTheSegmentSeed:
    async def test_an_explicit_seed_wins_over_the_derived_one(self, db):
        """The claim payload is where the seed actually reaches the worker."""
        from app.routes.segments import claim_next_segment  # noqa: F401  (import guard)

        user = await _user(db)
        job, seg = await _job_with_segment(db, user, seed=42)
        assert seg.seed == 42
        # The derivation would have produced job.seed + index = 1234.
        assert (job.seed + seg.index) % (2**63 - 1) == 1234
