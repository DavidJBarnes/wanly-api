"""A worker that re-registers holds nothing, so its old claims must be freed.

THE INCIDENT (2026-09-06). The 3090's container was replaced 50% through a 673 MB LoRA
download in [2/6]. The replacement registered under the same friendly_name, so the worker row
-- and its id -- was REUSED, it heartbeated immediately, and it claimed something else.

The segment it had been working on sat in PROCESSING for SEVEN HOURS, pinned to a worker that
was alive, healthy, busy, and had never heard of it. Its job could not finish, and the console
showed two segments in progress against one worker.

NOTHING EXISTING COULD FREE IT. The reclaim in /segments/next needs either a stale heartbeat
or an idle worker with an empty progress log. A replaced container satisfies neither.
"""
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import verify_api_key
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User, Worker

pytestmark = pytest.mark.asyncio


async def _user(db):
    u = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(u); await db.flush(); return u


async def _job(db, user):
    j = Job(user_id=user.id, name="j", width=832, height=1216, fps=24, seed=1,
            priority=0, status=JobStatus.PROCESSING, starting_image="s3://b/s.png")
    db.add(j); await db.flush(); return j


async def _worker(db, name="3090.zero", status="online-busy"):
    w = Worker(friendly_name=name, hostname="h", ip_address="10.0.0.1",
               status=status, comfyui_running=True)
    db.add(w); await db.flush(); return w


async def _segment(db, job, worker, status=SegmentStatus.PROCESSING, log="[2/6] 50% of 673 MB", index=0):
    from datetime import datetime, timezone
    s = Segment(job_id=job.id, index=index, prompt="p", status=status, discarded=False,
                worker_id=worker.id if worker else None,
                worker_name=worker.friendly_name if worker else None,
                claimed_at=datetime.now(timezone.utc), progress_log=log)
    db.add(s); await db.flush(); return s


async def _register(db, name="3090.zero"):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        for obj in list(db.identity_map.values()):
            if isinstance(obj, (Job, Segment, Worker)):
                db.expire(obj)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.post("/workers", json={
                "friendly_name": name, "hostname": "h",
                "ip_address": "10.0.0.1", "comfyui_running": True,
            })
    finally:
        app.dependency_overrides.clear()


async def test_a_reregistering_worker_releases_the_claim_it_abandoned(db):
    """THE regression. Seven hours stuck, and unreachable by every existing rule."""
    w = await _worker(db)
    seg = await _segment(db, await _job(db, await _user(db)), w)

    assert (await _register(db)).status_code == 201

    await db.refresh(seg)
    assert seg.status == SegmentStatus.PENDING
    assert seg.worker_id is None and seg.worker_name is None
    assert seg.claimed_at is None


async def test_the_stale_progress_log_goes_with_it(db):
    """It describes an attempt that no longer exists — and leaving it would make the segment
    permanently ineligible for the live-worker reclaim in /segments/next, which requires an
    empty log."""
    w = await _worker(db)
    seg = await _segment(db, await _job(db, await _user(db)), w)
    await _register(db)
    await db.refresh(seg)
    assert not (seg.progress_log or "")


async def test_a_claimed_but_not_yet_processing_segment_is_freed_too(db):
    """The window this incident happened in — [1/6]/[2/6], before the engine ever sees it."""
    w = await _worker(db)
    seg = await _segment(db, await _job(db, await _user(db)), w,
                         status=SegmentStatus.CLAIMED)
    await _register(db)
    await db.refresh(seg)
    assert seg.status == SegmentStatus.PENDING


async def test_finished_work_is_left_alone(db):
    """Only in-flight claims. Rewriting a completed segment would destroy a real result."""
    w = await _worker(db)
    job = await _job(db, await _user(db))
    done = await _segment(db, job, w, status=SegmentStatus.COMPLETED, log="[6/6] done", index=0)
    failed = await _segment(db, job, w, status=SegmentStatus.FAILED, log="boom", index=1)
    await _register(db)
    await db.refresh(done); await db.refresh(failed)
    assert done.status == SegmentStatus.COMPLETED
    assert done.worker_id == w.id and done.progress_log == "[6/6] done"
    assert failed.status == SegmentStatus.FAILED


async def test_another_workers_claim_is_untouched(db):
    """Scoped to the registering worker. Freeing someone else's in-flight segment would hand
    live work to a second GPU — the exact convergence this must not cause."""
    await _worker(db, "3090.zero")
    theirs = await _worker(db, "runpod-1")
    job = await _job(db, await _user(db))
    other = await _segment(db, job, theirs)
    # Read the id BEFORE the request expires the object: touching an expired attribute
    # outside the route's greenlet triggers a sync lazy load and raises MissingGreenlet.
    theirs_id = theirs.id

    await _register(db, "3090.zero")

    await db.refresh(other)
    assert other.status == SegmentStatus.PROCESSING
    assert other.worker_id == theirs_id


async def test_a_brand_new_worker_registers_cleanly(db):
    """No prior row, nothing to release, must not error."""
    assert (await _register(db, f"fresh-{uuid.uuid4().hex[:6]}")).status_code == 201
