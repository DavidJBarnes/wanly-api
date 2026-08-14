"""Migration 066: archived takes that predate seed recording get the seed they ran on.

Executed against a real database rather than asserted as a string, because the two things that
can go wrong are both database behaviours: bigint overflow on a large job seed, and touching rows
the migration is supposed to leave alone.
"""

import importlib.util
import pathlib
import uuid

import pytest
from sqlalchemy import select, text

from app.enums import JobStatus, SegmentStatus
from app.models import Job, Segment, User

_spec = importlib.util.spec_from_file_location(
    "migration_066",
    pathlib.Path(__file__).resolve().parents[1]
    / "alembic/versions/066_backfill_archived_take_seeds.py",
)
_migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migration)
BACKFILL_SQL = _migration.BACKFILL_SQL


async def _job(db, seed: int) -> Job:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    job = Job(
        user_id=user.id, name="j", width=480, height=832, fps=16, seed=seed,
        status=JobStatus.AWAITING,
    )
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, index: int, discarded: bool, seed: int | None = None) -> Segment:
    seg = Segment(
        job_id=job.id, index=index, prompt="p", status=SegmentStatus.COMPLETED,
        discarded=discarded, seed=seed,
    )
    db.add(seg)
    await db.flush()
    return seg


@pytest.mark.asyncio
class TestBackfill:
    async def test_an_archived_take_gets_the_seed_it_ran_on(self, db):
        # Until a segment carried its own seed, the worker was handed job.seed + index.
        job = await _job(db, 1000)
        seg = await _segment(db, job, index=2, discarded=True)

        await db.execute(text(BACKFILL_SQL))

        await db.refresh(seg)
        assert seg.seed == 1002

    async def test_a_live_segment_is_left_alone(self, db):
        """Live segments keep deriving. NULL there means "never asked for a particular seed",
        which is the honest record while the segment still holds its position."""
        job = await _job(db, 1000)
        seg = await _segment(db, job, index=0, discarded=False)

        await db.execute(text(BACKFILL_SQL))

        await db.refresh(seg)
        assert seg.seed is None

    async def test_an_already_recorded_seed_is_not_overwritten(self, db):
        # A take archived after stamping shipped already carries the truth.
        job = await _job(db, 1000)
        seg = await _segment(db, job, index=0, discarded=True, seed=150488800771430)

        await db.execute(text(BACKFILL_SQL))

        await db.refresh(seg)
        assert seg.seed == 150488800771430

    async def test_a_seed_at_the_top_of_the_range_does_not_overflow(self, db):
        """95% of job seeds came from the full 2**63 range, so job.seed + index can exceed what
        bigint holds. Postgres raises on overflow rather than wrapping, which would abort the
        entire migration — hence the numeric cast."""
        job = await _job(db, 9223372036854775800)
        seg = await _segment(db, job, index=7, discarded=True)

        await db.execute(text(BACKFILL_SQL))

        await db.refresh(seg)
        # (9223372036854775800 + 7) % (2**63 - 1) == 0
        assert seg.seed == (9223372036854775800 + 7) % (2**63 - 1)

    async def test_it_matches_what_the_reroll_endpoint_writes(self, db):
        # Two code paths, one meaning. If they diverge, the same clip gets two different answers.
        job = await _job(db, 4611686018427387904)
        seg = await _segment(db, job, index=3, discarded=True)

        await db.execute(text(BACKFILL_SQL))

        await db.refresh(seg)
        assert seg.seed == (job.seed + seg.index) % (2**63 - 1)

    async def test_it_does_not_reach_across_jobs(self, db):
        job_a = await _job(db, 100)
        job_b = await _job(db, 999999)
        seg = await _segment(db, job_a, index=1, discarded=True)
        await _segment(db, job_b, index=1, discarded=True)

        await db.execute(text(BACKFILL_SQL))

        await db.refresh(seg)
        assert seg.seed == 101

    async def test_running_it_twice_changes_nothing(self, db):
        # Alembic will not, but a hand re-run during a repair might.
        job = await _job(db, 1000)
        seg = await _segment(db, job, index=0, discarded=True)

        await db.execute(text(BACKFILL_SQL))
        await db.refresh(seg)
        first = seg.seed
        await db.execute(text(BACKFILL_SQL))
        await db.refresh(seg)

        assert seg.seed == first == 1000

    async def test_every_archived_take_ends_up_with_a_seed(self, db):
        job = await _job(db, 1000)
        for i in range(3):
            await _segment(db, job, index=i, discarded=True)

        await db.execute(text(BACKFILL_SQL))

        rows = (
            await db.execute(
                select(Segment.seed).where(Segment.job_id == job.id, Segment.discarded)
            )
        ).scalars().all()
        assert all(s is not None for s in rows)
