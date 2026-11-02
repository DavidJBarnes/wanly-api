"""GET /holograms: the pool for the shuffled AR playlist (console#713).

Real Postgres, because the behaviour is which rows the join and the job filters let through.
"""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import get_current_user
from app.database import get_db
from app.enums import JobStatus, SegmentStatus
from app.main import app
from app.models import Job, Segment, User


async def _user(db):
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    return user


async def _hologram(db, user, name, tags=None, status=JobStatus.FINALIZED, paths=True):
    job = Job(user_id=user.id, name=name, tags=tags, status=status,
              width=640, height=640, fps=30, seed=1)
    db.add(job)
    await db.flush()
    db.add(Segment(
        job_id=job.id, index=1000, prompt="", duration_seconds=5,
        status=SegmentStatus.COMPLETED,
        hologram_video_path=f"s3://b/{job.id}.mp4" if paths else None,
        hologram_manifest_path=f"s3://b/{job.id}.json" if paths else None,
    ))
    await db.flush()
    return job


async def _names(db, user, **params):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            res = await c.get("/holograms", params=params)
    finally:
        app.dependency_overrides.clear()
    assert res.status_code == 200
    return {h["job_name"] for h in res.json()}


@pytest.mark.asyncio
async def test_only_finished_holograms_on_finalized_jobs(db):
    user = await _user(db)
    await _hologram(db, user, "ready")
    await _hologram(db, user, "remaking", paths=False)
    await _hologram(db, user, "not finalized", status=JobStatus.PROCESSING)
    assert await _names(db, user) == {"ready"}


@pytest.mark.asyncio
async def test_other_users_holograms_are_not_listed(db):
    user, other = await _user(db), await _user(db)
    await _hologram(db, other, "theirs")
    assert await _names(db, user) == set()


@pytest.mark.asyncio
async def test_honours_the_job_list_filters(db):
    user = await _user(db)
    await _hologram(db, user, "kitchen take", tags="kelly, ar")
    await _hologram(db, user, "driveway take", tags="ar")
    assert await _names(db, user, tags=["kelly"]) == {"kitchen take"}
    assert await _names(db, user, q="driveway") == {"driveway take"}


@pytest.mark.asyncio
async def test_row_carries_what_the_player_needs(db):
    user = await _user(db)
    job = await _hologram(db, user, "ready")
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            row = (await c.get("/holograms")).json()[0]
    finally:
        app.dependency_overrides.clear()
    assert row["job_id"] == str(job.id)
    assert row["flavor"] == "2d_matte"
    assert row["video_path"].endswith(".mp4") and row["manifest_path"].endswith(".json")
