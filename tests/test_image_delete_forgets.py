"""Deleting an image drops what the database says about it (console#559).

An image's own caption and tags live on its image_meta row; each dataset holding it keeps a
training caption and a likeness score under its URI. Removing an image from a set and
cropping already pruned the dataset half. Deleting the file itself left all of it behind.

Against a real Postgres: the dataset half is a JSONB ?| match and a FOR UPDATE, neither of
which a fake session would exercise.
"""
import asyncio
import uuid
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Dataset, ImageMeta

BUCKET = "wanly-images"


def _uri(name: str) -> str:
    return f"s3://{BUCKET}/forget-{uuid.uuid4().hex[:8]}/{name}.png"


@pytest_asyncio.fixture
async def api(db):
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: object()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)


async def _delete(api, path, force=False):
    params = {"path": path, **({"force": "true"} if force else {})}
    with patch("app.routes.images.delete_object") as deleter:
        resp = await api.delete("/images", params=params)
    return resp, deleter


class TestDeleteForgetsTheImage:
    async def test_its_caption_and_tags_row_goes(self, db, api):
        path = _uri("a")
        db.add(ImageMeta(path=path, tags="kelly", scene_description="a woman on a sofa"))
        await db.commit()

        resp, deleter = await _delete(api, path)

        assert resp.status_code == 200
        deleter.assert_called_once_with(path)
        assert await db.get(ImageMeta, path) is None

    async def test_dataset_caption_and_score_entries_go_membership_stays(self, db, api):
        gone, kept = _uri("gone"), _uri("kept")
        ds = Dataset(name=f"forget-{uuid.uuid4().hex[:6]}", images=[gone, kept],
                     prefix="datasets/p",
                     captions={gone: "close-up, smiling", kept: "full body"},
                     scores={gone: 0.61, kept: 0.72})
        db.add(ds)
        await db.commit()

        resp, _ = await _delete(api, gone, force=True)

        assert resp.status_code == 200
        await db.refresh(ds)
        assert ds.captions == {kept: "full body"}
        assert ds.scores == {kept: 0.72}
        # The dead entry the 409 dialog warned about: membership is the set's own business.
        assert ds.images == [gone, kept]

    async def test_a_set_that_never_mentioned_it_is_untouched(self, db, api):
        other = _uri("other")
        ds = Dataset(name=f"forget-{uuid.uuid4().hex[:6]}", images=[other],
                     prefix="datasets/p", captions={other: "x"}, scores={other: 0.5})
        db.add(ds)
        await db.commit()

        resp, _ = await _delete(api, _uri("unrelated"))

        assert resp.status_code == 200
        await db.refresh(ds)
        assert ds.captions == {other: "x"} and ds.scores == {other: 0.5}



@pytest_asyncio.fixture
async def committed(db_engine):
    """The app on real, separately committed sessions -- what production does -- for the
    tests that need a second connection or a real rollback. Cleans up its own rows."""
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    maker = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with maker() as s:
            yield s

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user] = lambda: object()
    made: list = []
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            yield maker, c, made
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        async with maker() as s:
            for obj in made:
                await s.execute(delete(type(obj)).where(
                    *[col == getattr(obj, col.key) for col in type(obj).__table__.primary_key]))
            await s.commit()


async def _commit(maker, made, obj):
    async with maker() as s:
        s.add(obj)
        await s.commit()
    made.append(obj)
    return obj


class TestDeleteIsAllOrNothing:
    async def test_a_failed_s3_delete_keeps_the_caption(self, committed):
        maker, api, made = committed
        path = _uri("stuck")
        await _commit(maker, made, ImageMeta(path=path, scene_description="still here"))

        with patch("app.routes.images.delete_object", side_effect=RuntimeError("S3 down")):
            try:
                await api.delete("/images", params={"path": path})
            except RuntimeError:
                pass  # ASGITransport re-raises the unhandled error; production answers 500
        async with maker() as s:
            assert (await s.get(ImageMeta, path)).scene_description == "still here"


class TestDeleteFailsFast:
    async def test_a_held_dataset_row_is_a_prompt_503_not_a_hang(self, committed, monkeypatch):
        """Nothing should hold these rows for long; if something ever does, the delete says
        so within DELETE_LOCK_TIMEOUT instead of spinning."""
        from app.routes import images

        maker, api, made = committed
        monkeypatch.setattr(images, "DELETE_LOCK_TIMEOUT", "300ms")
        path = _uri("held")
        ds = await _commit(maker, made, Dataset(
            name=f"forget-{uuid.uuid4().hex[:6]}", images=[path], prefix="datasets/p",
            captions={path: "x"}, scores={}))
        async with maker() as holder:
            await holder.execute(select(Dataset).where(Dataset.id == ds.id).with_for_update())
            with patch("app.routes.images.delete_object") as deleter:
                resp = await asyncio.wait_for(
                    api.delete("/images", params={"path": path, "force": "true"}), timeout=5)
            await holder.rollback()
        assert resp.status_code == 503
        assert "being written" in resp.json()["detail"]
        deleter.assert_not_called()
        async with maker() as s:
            assert (await s.get(Dataset, ds.id)).captions == {path: "x"}
