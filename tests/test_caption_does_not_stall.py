"""Captioning must not stall the rest of the API (console#559).

WHAT HAPPENED. Every request's first query is the auth lookup, which opens the session's
transaction and checks a connection out of the pool. POST /images/scene then waited for its
turn in the caption queue -- minutes, behind a dataset's captioning -- still holding that
connection, "idle in transaction". Enough waiting describes and the pool (5 + 10) was empty:
every other request, a delete or a listing, sat on the pool behind captioning and then died
with an unhandled TimeoutError.

These tests run the real routes against a real Postgres with a deliberately tiny pool, so
"enough" is two, and a captioner that does not answer until the test lets it. The rest of
the suite's `db` fixture cannot show this: it binds every session to one connection.

Rows here are COMMITTED (the whole point is separate connections), so each test removes
what it wrote.
"""
import asyncio
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Dataset, ImageMeta, User

BUCKET = "wanly-images"
POOL = 2


@pytest_asyncio.fixture
async def small_pool(db_engine):
    """A sessionmaker over a 2-connection pool, wired in as the app's get_db."""
    from app.database import get_db
    from app.main import app
    from tests.conftest import _test_database_url

    engine = create_async_engine(_test_database_url(), pool_size=POOL, max_overflow=0,
                                 pool_timeout=3)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_db] = _get_db
    try:
        yield maker
    finally:
        app.dependency_overrides.pop(get_db, None)
        await engine.dispose()


@pytest_asyncio.fixture
async def user(small_pool):
    from app.auth import create_access_token

    uid = uuid.uuid4()
    async with small_pool() as s:
        s.add(User(id=uid, username=f"stall-{uid.hex[:8]}", password_hash="x"))
        await s.commit()
    yield {"Authorization": f"Bearer {create_access_token(uid)}"}
    async with small_pool() as s:
        await s.execute(delete(User).where(User.id == uid))
        await s.commit()


@pytest.fixture
def captioner(monkeypatch, small_pool):
    """A captioner that answers only when `release` is set, and no S3."""
    from app import caption_queue, caption_tickets, config, s3
    from app.routes import captions, images

    release = asyncio.Event()
    # A fresh queue per test: its asyncio.Lock binds to the loop that first waits on it, and
    # pytest-asyncio gives each test its own.
    monkeypatch.setattr(caption_queue, "queue", caption_queue.CaptionQueue())

    async def describe(image, instruction, base_url=None):
        await release.wait()
        return "a caption"

    async def not_busy(db):
        return None

    async def no_settings(db):
        return {}

    monkeypatch.setattr(captions, "describe", describe)
    monkeypatch.setattr(captions, "busy_render_beside_the_captioner", not_busy)
    monkeypatch.setattr(captions, "_get_all_settings", no_settings)
    monkeypatch.setattr(config.settings, "motion_caption_enabled", False)
    monkeypatch.setattr(images, "download_bytes", lambda path: b"img")
    monkeypatch.setattr(s3, "download_bytes", lambda path: b"img")
    monkeypatch.setattr(images, "delete_object", lambda path: None)
    # A describe is a caption ticket since console#564: its short sessions come from the same
    # tiny pool, which is the point -- they must not pile up on it either.
    monkeypatch.setattr(caption_tickets, "async_session", small_pool)
    caption_tickets.reset()
    yield release
    caption_tickets.reset()


def _path(tag: str) -> str:
    return f"s3://{BUCKET}/stall-{tag}/{uuid.uuid4().hex[:8]}.png"


async def _cleanup(maker, paths):
    async with maker() as s:
        await s.execute(delete(ImageMeta).where(ImageMeta.path.in_(paths)))
        await s.commit()


async def _queue_describes(client, headers, paths):
    tasks = [asyncio.create_task(client.post("/images/scene", params={"path": p},
                                             headers=headers)) for p in paths]
    # Let every one of them reach the caption queue (one captioning, the rest in line).
    for _ in range(50):
        await asyncio.sleep(0.02)
        from app.caption_queue import queue
        if queue.depth() >= len(paths):
            break
    return tasks


class TestOtherRequestsWhileCaptioning:
    async def test_a_delete_completes_while_describes_wait_their_turn(
            self, small_pool, user, captioner):
        """More describes in flight than the pool has connections -- the reported state."""
        from app.main import app

        waiting = [_path("wait") for _ in range(POOL + 1)]
        victim = _path("victim")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t",
                               timeout=30) as c:
            tasks = await _queue_describes(c, user, waiting)
            try:
                resp = await asyncio.wait_for(
                    c.delete("/images", params={"path": victim}, headers=user), timeout=2)
                assert resp.status_code == 200
                # Not only deletes: any request that needs the database gets through.
                resp = await asyncio.wait_for(
                    c.get("/images/scene", params={"path": victim}, headers=user), timeout=2)
                assert resp.status_code == 200
            finally:
                captioner.set()
                done = await asyncio.gather(*tasks)
                await _cleanup(small_pool, waiting)
        assert [r.status_code for r in done] == [200] * len(waiting)

    async def test_a_request_completes_while_a_dataset_captions(
            self, small_pool, user, captioner, monkeypatch):
        """The background caption loop, plus a describe queued behind it."""
        from app.main import app
        from app.routes import datasets as ds_mod

        class _S3:
            def download_bytes(self, uri):
                return b"img"

            def head_object(self, uri):
                return {"Key": uri}

        monkeypatch.setattr(ds_mod, "s3", _S3())
        monkeypatch.setattr(ds_mod, "async_session", small_pool)
        imgs = [_path("ds") for _ in range(3)]
        async with small_pool() as s:
            ds = Dataset(name=f"stall-{uuid.uuid4().hex[:6]}", images=imgs, prefix="datasets/p",
                         captions={}, scores={})
            s.add(ds)
            await s.commit()
        ds_id = ds.id
        extra = [_path("extra") for _ in range(POOL)]
        loop = asyncio.create_task(ds_mod.caption_dataset_job(ds_id, False))
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t",
                                   timeout=30) as c:
                tasks = await _queue_describes(c, user, extra)
                resp = await asyncio.wait_for(
                    c.get(f"/datasets/{ds_id}/captions/status", headers=user), timeout=2)
                assert resp.status_code == 200
                assert resp.json()["captioned"] == 0  # the captioner has not answered yet
                captioner.set()
                await asyncio.gather(*tasks)
            await asyncio.wait_for(loop, timeout=10)
            async with small_pool() as s:
                got = (await s.execute(select(Dataset).where(Dataset.id == ds_id))).scalar_one()
                assert set(got.captions) == set(imgs)
        finally:
            captioner.set()
            async with small_pool() as s:
                await s.execute(delete(Dataset).where(Dataset.id == ds_id))
                await s.commit()
            await _cleanup(small_pool, extra)
            ds_mod._CAPTION_RUNS.pop(ds_id, None)


class TestAnEmptyPoolIsA503:
    async def test_a_pool_timeout_says_so(self, monkeypatch):
        """If the pool ever does run dry, the answer is prompt and says why -- not a 500."""
        from sqlalchemy.exc import TimeoutError as PoolTimeout

        from app.auth import get_current_user
        from app.main import app

        async def _dry():
            raise PoolTimeout("QueuePool limit of size 5 overflow 10 reached")

        app.dependency_overrides[get_current_user] = _dry
        try:
            async with AsyncClient(transport=ASGITransport(app=app),
                                   base_url="http://t") as c:
                resp = await c.delete("/images", params={"path": f"s3://{BUCKET}/a.png"})
        finally:
            app.dependency_overrides.pop(get_current_user, None)
        assert resp.status_code == 503
        assert "database connections" in resp.json()["detail"]
