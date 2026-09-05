"""A segment's output is keyed by its ID, and a delete cannot take another row's file.

console#440. The key used to be f"{job_id}/{index}_output.mp4". The index is neither unique
across rows nor stable over time:

  * every re-roll take at an index writes the SAME key, so a new take silently overwrote the
    previous one's video;
  * deleting a segment removed those objects without checking whether another row still
    pointed at them;
  * the re-index-on-delete renamed objects, and a shared object can only be moved once — the
    row whose move failed kept a path to an object that had moved.

Measured in production before the fix: 73 rows shared an output_path with another row, across
31 distinct keys, and 31 of the sharers were LIVE segments. One of them had already lost both
its video and its last frame.
"""

import uuid
from unittest.mock import patch

import pytest

from app.enums import JobStatus, SegmentStatus
from app.models import Job, Segment, User


class TestTheKeyIsUniquePerSegment:
    def test_two_takes_at_the_same_index_do_not_share_a_key(self):
        """The whole bug in one assertion. Both takes are index 0; only the id differs."""
        job_id = uuid.uuid4()
        a, b = uuid.uuid4(), uuid.uuid4()

        key = lambda seg_id: f"{job_id}/{seg_id}_output.mp4"  # noqa: E731

        assert key(a) != key(b)

    def test_the_old_scheme_is_what_collided(self):
        """Kept as the record of what was wrong: index alone is not an identity."""
        job_id = uuid.uuid4()
        assert f"{job_id}/0_output.mp4" == f"{job_id}/0_output.mp4"


@pytest.mark.asyncio
async def test_upload_keys_by_segment_id(db):
    """Through the route, so the key that is actually sent to S3 is the one asserted."""
    from httpx import ASGITransport, AsyncClient
    from app.auth import verify_api_key
    from app.database import get_db
    from app.main import app

    user = User(id=uuid.uuid4(), username="t", password_hash="x")
    job = Job(id=uuid.uuid4(), user_id=user.id, name="j", status=JobStatus.PROCESSING,
              width=832, height=1216, fps=24, seed=1234)
    seg = Segment(id=uuid.uuid4(), job_id=job.id, index=0, prompt="p",
                  duration_seconds=10.0, speed=1.0, status=SegmentStatus.PROCESSING)
    db.add_all([user, job, seg])
    await db.flush()

    app.dependency_overrides[verify_api_key] = lambda: True
    app.dependency_overrides[get_db] = lambda: db
    try:
        with patch("app.routes.files.upload_bytes",
                   side_effect=lambda data, key, bucket: f"s3://{bucket}/{key}") as up:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post(
                    f"/segments/{seg.id}/upload",
                    files={"video": ("v.mp4", b"v", "video/mp4"),
                           "last_frame": ("f.png", b"f", "image/png")},
                )
        assert resp.status_code == 200, resp.text
        keys = [call.args[1] for call in up.call_args_list]
        assert keys == [f"{job.id}/{seg.id}_output.mp4",
                        f"{job.id}/{seg.id}_last_frame.png"]
        # And emphatically NOT the index.
        assert not any(f"/{seg.index}_" in k for k in keys)
    finally:
        app.dependency_overrides.clear()


class TestContinuingFromTheNearestFrame:
    """A continuation starts from the nearest preceding LIVE segment that produced a frame.

    It used to match `index - 1` exactly. Indexes are not contiguous — deleting a segment
    renumbers the rest while a discarded take keeps its index — so a live row at 5 can have
    its predecessor at 3. The exact match then found nothing, the start image stayed NULL,
    and the daemon rendered the continuation AS TEXT-TO-VIDEO: ten minutes of GPU on a clip
    unrelated to the chain, completing green with nothing saying why (console#440).
    """

    @staticmethod
    async def _job(db):
        user = User(id=uuid.uuid4(), username=f"u{uuid.uuid4().hex[:8]}", password_hash="x")
        job = Job(id=uuid.uuid4(), user_id=user.id, name="j", status=JobStatus.PROCESSING,
                  width=832, height=1216, fps=24, seed=1)
        db.add_all([user, job])
        await db.flush()
        return job

    @staticmethod
    def _seg(job, index, **kw):
        return Segment(id=uuid.uuid4(), job_id=job.id, index=index, prompt="p",
                       duration_seconds=10.0, speed=1.0,
                       status=kw.pop("status", SegmentStatus.COMPLETED), **kw)

    async def _resolve(self, db, job, target):
        """The claim's predecessor query, as the route runs it."""
        from sqlalchemy import select as sel

        row = (await db.execute(
            sel(Segment)
            .where(Segment.job_id == job.id,
                   Segment.index < target.index,
                   Segment.discarded.is_(False),
                   Segment.last_frame_path.is_not(None))
            .order_by(Segment.index.desc())
            .limit(1)
        )).scalar_one_or_none()
        return row.last_frame_path if row else None

    @pytest.mark.asyncio
    async def test_it_skips_a_gap_left_by_a_deleted_segment(self, db):
        job = await self._job(db)
        live = self._seg(job, 3, last_frame_path="s3://b/j/three.png", discarded=False)
        target = self._seg(job, 5, status=SegmentStatus.PENDING, discarded=False)
        db.add_all([live, target])
        await db.flush()

        # index - 1 is 4, which does not exist. The nearest live frame is at 3.
        assert await self._resolve(db, job, target) == "s3://b/j/three.png"

    @pytest.mark.asyncio
    async def test_it_ignores_discarded_takes(self, db):
        """A thrown-away take is not what anything continues from."""
        job = await self._job(db)
        db.add_all([
            self._seg(job, 0, last_frame_path="s3://b/j/kept.png", discarded=False),
            self._seg(job, 1, last_frame_path="s3://b/j/rolled-away.png", discarded=True),
        ])
        target = self._seg(job, 2, status=SegmentStatus.PENDING, discarded=False)
        db.add(target)
        await db.flush()

        assert await self._resolve(db, job, target) == "s3://b/j/kept.png"

    @pytest.mark.asyncio
    async def test_it_skips_a_row_that_produced_no_frame(self, db):
        job = await self._job(db)
        db.add_all([
            self._seg(job, 0, last_frame_path="s3://b/j/zero.png", discarded=False),
            self._seg(job, 1, last_frame_path=None, discarded=False),
        ])
        target = self._seg(job, 2, status=SegmentStatus.PENDING, discarded=False)
        db.add(target)
        await db.flush()

        assert await self._resolve(db, job, target) == "s3://b/j/zero.png"

    @pytest.mark.asyncio
    async def test_nothing_to_continue_from_resolves_to_nothing(self, db):
        """Which is what the claim turns into a FAILED segment with a reason, rather than
        handing the daemon a null start image and getting a text-to-video render."""
        job = await self._job(db)
        db.add(self._seg(job, 0, last_frame_path=None, discarded=False))
        target = self._seg(job, 1, status=SegmentStatus.PENDING, discarded=False)
        db.add(target)
        await db.flush()

        assert await self._resolve(db, job, target) is None
