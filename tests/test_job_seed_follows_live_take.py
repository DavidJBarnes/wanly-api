"""One live take, one seed, and it lives on the job.

Re-roll used to write the new seed onto the segment and leave job.seed alone, so a re-rolled job
displayed two seeds — the job header showing the seed of a take that had already been archived,
and segment 0 showing the one actually in use. Two numbers, the more prominent one wrong.

The job is where the seed belongs: the header shows it and the create dialog accepts it back to
reproduce a job. Segment 0 derives from it like every other segment.

This is only safe because archiving stamps the outgoing take with the seed it ran on. Overwriting
job.seed without that would relabel an archived clip with a seed that never generated it — which
is why that stamping and this move cannot be separated.
"""

import importlib.util
import pathlib
import uuid

import pytest
from sqlalchemy import select, text

from app.auth import get_current_user
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User

_spec = importlib.util.spec_from_file_location(
    "migration_067",
    pathlib.Path(__file__).resolve().parents[1]
    / "alembic/versions/067_job_seed_follows_the_live_take.py",
)
_migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migration)


async def _run_migration(db):
    await db.execute(text(_migration.MOVE_SEED_TO_JOB))
    await db.execute(text(_migration.CLEAR_SEED_ON_LIVE_SEGMENTS))


async def _user(db) -> User:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _job(db, user=None, *, seed=1000) -> Job:
    user = user or await _user(db)
    job = Job(
        user_id=user.id, name="j", width=480, height=832, fps=16, seed=seed,
        starting_image="s3://b/start.png", status=JobStatus.AWAITING,
    )
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, *, index=0, discarded=False, seed=None,
                   status=SegmentStatus.COMPLETED) -> Segment:
    seg = Segment(job_id=job.id, index=index, prompt="p", status=status,
                  discarded=discarded, seed=seed)
    db.add(seg)
    await db.flush()
    return seg


async def _reroll(db, user, job_id):
    from httpx import ASGITransport, AsyncClient

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
class TestRerollMovesTheSeedToTheJob:
    async def test_the_job_gets_the_new_seed(self, db):
        user = await _user(db)
        job = await _job(db, user, seed=1000)
        await _segment(db, job)

        await _reroll(db, user, job.id)

        await db.refresh(job)
        assert job.seed != 1000

    async def test_the_new_take_carries_no_seed_of_its_own(self, db):
        # One number, one place. A copy on the segment is the same value twice, free to drift.
        user = await _user(db)
        job = await _job(db, user, seed=1000)
        await _segment(db, job)

        body = (await _reroll(db, user, job.id)).json()

        assert body["seed"] is None

    async def test_the_archived_take_keeps_the_seed_it_ran_on(self, db):
        # The whole reason overwriting job.seed is safe.
        user = await _user(db)
        job = await _job(db, user, seed=1000)
        old = await _segment(db, job)

        await _reroll(db, user, job.id)

        await db.refresh(old)
        assert old.seed == 1000

    async def test_the_new_take_claims_the_seed_now_on_the_job(self, db):
        """End to end: what the worker is handed must be the number the header shows."""
        user = await _user(db)
        job = await _job(db, user, seed=1000)
        await _segment(db, job)

        await _reroll(db, user, job.id)
        await db.refresh(job)

        fresh = (
            await db.execute(
                select(Segment).where(Segment.job_id == job.id, Segment.discarded.is_(False))
            )
        ).scalar_one()
        # The claim derivation for a NULL-seed segment at index 0.
        assert fresh.seed is None
        assert (job.seed + fresh.index) % (2**63 - 1) == job.seed

    async def test_rolling_twice_leaves_two_distinct_archived_seeds(self, db):
        # Each archived take answers for itself; the job answers for the live one.
        user = await _user(db)
        job = await _job(db, user, seed=1000)
        await _segment(db, job)

        await _reroll(db, user, job.id)
        await db.refresh(job)
        second_seed = job.seed
        live = (
            await db.execute(
                select(Segment).where(Segment.job_id == job.id, Segment.discarded.is_(False))
            )
        ).scalar_one()
        live.status = SegmentStatus.COMPLETED
        await db.flush()

        await _reroll(db, user, job.id)
        await db.refresh(job)

        archived = (
            await db.execute(
                select(Segment.seed).where(Segment.job_id == job.id, Segment.discarded)
            )
        ).scalars().all()
        assert sorted(archived) == sorted([1000, second_seed])
        assert job.seed not in archived


@pytest.mark.asyncio
class TestMigration067:
    async def test_it_moves_a_live_segments_seed_onto_the_job(self, db):
        job = await _job(db, seed=7189671993964474770)          # the old take's seed
        live = await _segment(db, job, seed=5699167371037795)   # the take actually in use
        await _segment(db, job, discarded=True, seed=7189671993964474770)

        await _run_migration(db)

        await db.refresh(job)
        await db.refresh(live)
        assert job.seed == 5699167371037795
        assert live.seed is None

    async def test_the_archived_take_is_untouched(self, db):
        job = await _job(db, seed=1000)
        await _segment(db, job, seed=2000)
        archived = await _segment(db, job, discarded=True, seed=1000)

        await _run_migration(db)

        await db.refresh(archived)
        assert archived.seed == 1000

    async def test_a_job_that_was_never_re_rolled_is_untouched(self, db):
        job = await _job(db, seed=1000)
        live = await _segment(db, job)

        await _run_migration(db)

        await db.refresh(job)
        await db.refresh(live)
        assert job.seed == 1000 and live.seed is None

    async def test_a_live_seed_with_no_archived_sibling_is_left_alone(self, db):
        """A shape this has not seen. Moving the seed there would drop the old job seed with
        nothing else recording it, so it is left for a human rather than guessed at."""
        job = await _job(db, seed=1000)
        live = await _segment(db, job, seed=2000)

        await _run_migration(db)

        await db.refresh(job)
        await db.refresh(live)
        assert job.seed == 1000
        assert live.seed == 2000

    async def test_running_it_twice_changes_nothing(self, db):
        job = await _job(db, seed=1000)
        live = await _segment(db, job, seed=2000)
        await _segment(db, job, discarded=True, seed=1000)

        await _run_migration(db)
        await _run_migration(db)

        await db.refresh(job)
        await db.refresh(live)
        assert job.seed == 2000 and live.seed is None
