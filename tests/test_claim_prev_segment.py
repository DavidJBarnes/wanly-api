"""Claiming a segment must survive a job that has been re-rolled.

Segment index stopped identifying one row when re-rolling arrived: an archived take keeps its
index so the record reads as "the discarded version of segment 0", and its replacement takes the
same index as its position in the video. The claim endpoint resolved "the previous segment" by
index alone, so a job rolled six times had seven rows at index 0 and the lookup raised
MultipleResultsFound — a 500 on EVERY claim, which stops the queue dead rather than degrading.

Live is also the right answer on its own terms: the lookup resolves the frame the next segment
continues from, and a discarded take is not what anything continues from.
"""

import uuid

import pytest
from sqlalchemy import select

from app.enums import JobStatus, SegmentStatus
from app.models import Job, Segment, User


async def _job(db) -> Job:
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    job = Job(
        user_id=user.id, name="j", width=480, height=832, fps=16, seed=1000,
        starting_image="s3://b/start.png", status=JobStatus.PROCESSING,
    )
    db.add(job)
    await db.flush()
    return job


async def _segment(db, job, index, discarded=False, last_frame=None, status=SegmentStatus.COMPLETED):
    seg = Segment(
        job_id=job.id, index=index, prompt="p", status=status, discarded=discarded,
        last_frame_path=last_frame,
    )
    db.add(seg)
    await db.flush()
    return seg


async def _previous_segment(db, job, index: int):
    """Exactly the lookup the claim endpoint performs."""
    return (
        await db.execute(
            select(Segment).where(
                Segment.job_id == job.id,
                Segment.index == index - 1,
                Segment.discarded.is_(False),
            )
        )
    ).scalar_one_or_none()


@pytest.mark.asyncio
class TestPreviousSegmentLookup:
    async def test_it_survives_a_position_with_several_archived_takes(self, db):
        # The production failure: six re-rolls left seven rows at index 0 and every claim 500'd.
        job = await _job(db)
        for _ in range(6):
            await _segment(db, job, 0, discarded=True)
        live = await _segment(db, job, 0, last_frame="s3://b/live.png")
        await _segment(db, job, 1, status=SegmentStatus.PENDING)

        prev = await _previous_segment(db, job, 1)

        assert prev is not None and prev.id == live.id

    async def test_it_resolves_the_live_take_not_an_archived_one(self, db):
        # The frame a continuation starts from has to be the frame that is in the video.
        job = await _job(db)
        await _segment(db, job, 0, discarded=True, last_frame="s3://b/thrown-away.png")
        await _segment(db, job, 0, last_frame="s3://b/kept.png")

        prev = await _previous_segment(db, job, 1)

        assert prev.last_frame_path == "s3://b/kept.png"

    async def test_a_position_whose_take_was_discarded_resolves_to_nothing(self, db):
        # Not an error: the claim path treats a missing predecessor as "no start image to
        # inherit", which is the honest state when nothing live precedes this segment.
        job = await _job(db)
        await _segment(db, job, 0, discarded=True)

        assert await _previous_segment(db, job, 1) is None

    async def test_the_ordinary_single_take_case_is_unchanged(self, db):
        job = await _job(db)
        zero = await _segment(db, job, 0, last_frame="s3://b/0.png")

        assert (await _previous_segment(db, job, 1)).id == zero.id
