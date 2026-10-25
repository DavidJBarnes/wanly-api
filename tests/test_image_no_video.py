"""The Image Repo's No video filter (wanly-console#690).

"Show me every Kelly image I have never rendered." `no_video` narrows a search to images no job
or segment points at -- the same definition as the green "Used in a job" dot -- and ANDs with
the tag pills and the search box. On its own it scans the bucket like Untagged, because a
never-tagged image has no image_meta row and a query over that table would never find it.
"""

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.enums import JobStatus
from app.models import ImageMeta, Job, Segment, User
from app.routes.images import image_filter, no_video_clause

BUCKET = "wanly-images"


def _p(name):
    return f"s3://{BUCKET}/2026-10-01/{name}"


async def _job(db, starting_image=None, status=JobStatus.FINALIZED):
    user = User(username=str(uuid.uuid4()), password_hash="x")
    db.add(user)
    await db.flush()
    job = Job(user_id=user.id, name="j", status=status, starting_image=starting_image,
              width=640, height=640, fps=16, seed=1)
    db.add(job)
    await db.flush()
    return job


@pytest.mark.asyncio
class TestNoVideoClause:
    async def _paths(self, db, tags=()):
        clauses = image_filter(None, list(tags), []) + [no_video_clause()]
        rows = (await db.execute(select(ImageMeta.path).where(*clauses))).scalars().all()
        return {p.rsplit("/", 1)[1] for p in rows}

    async def test_a_start_image_drops_out(self, db):
        db.add_all([ImageMeta(path=_p("used.png"), tags="Kelly"),
                    ImageMeta(path=_p("fresh.png"), tags="Kelly")])
        await _job(db, starting_image=_p("used.png"))
        assert await self._paths(db, tags=["Kelly"]) == {"fresh.png"}

    async def test_it_ands_with_the_tag_pills(self, db):
        # The use case: Kelly images without a video, not every image without one.
        db.add_all([ImageMeta(path=_p("k.png"), tags="Kelly"),
                    ImageMeta(path=_p("g.png"), tags="Gena")])
        assert await self._paths(db, tags=["Kelly"]) == {"k.png"}

    async def test_a_segment_start_image_counts(self, db):
        # Same as the green dot: a segment pointing at the image means output exists.
        db.add(ImageMeta(path=_p("seg.png"), tags="Kelly"))
        job = await _job(db)
        db.add(Segment(job_id=job.id, index=0, prompt="p", start_image=_p("seg.png")))
        await db.flush()
        assert await self._paths(db, tags=["Kelly"]) == set()

    async def test_an_archived_job_still_counts(self, db):
        # "Have I rendered this" -- an archived render is still one, as it is for the dot.
        db.add(ImageMeta(path=_p("arch.png"), tags="Kelly"))
        await _job(db, starting_image=_p("arch.png"), status=JobStatus.ARCHIVED)
        assert await self._paths(db, tags=["Kelly"]) == set()


class _FakeS3:
    def __init__(self, keys):
        self._keys = keys  # key -> LastModified

    def list_common_prefixes(self, bucket):
        return sorted({k.split("/", 1)[0] + "/" for k in self._keys})

    def list_objects(self, bucket, prefix):
        return [{"Key": k, "Size": 1, "LastModified": m}
                for k, m in self._keys.items() if k.startswith(prefix)]

    def head_object(self, uri):
        key = uri.split("/", 3)[-1]
        m = self._keys.get(key)
        return {"Key": key, "Size": 1, "LastModified": m} if m else None


async def _get(db, keys, **params):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    fake = _FakeS3(keys)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    try:
        with patch("app.routes.images.list_common_prefixes", fake.list_common_prefixes), \
             patch("app.routes.images.list_objects", fake.list_objects), \
             patch("app.routes.images.head_object", fake.head_object):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                return await c.get("/images/search", params=params)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestSearchRoute:
    KEYS = {
        "2026-10-01/used.png": "2026-10-01T00:00:00Z",
        "2026-10-01/kelly.png": "2026-10-02T00:00:00Z",
        "2026-10-01/bare.png": "2026-10-03T00:00:00Z",   # no image_meta row at all
        "datasets/x/staging.png": "2026-10-04T00:00:00Z",
    }

    async def _seed(self, db):
        db.add_all([ImageMeta(path=_p("used.png"), tags="Kelly"),
                    ImageMeta(path=_p("kelly.png"), tags="Kelly")])
        await _job(db, starting_image=_p("used.png"))

    async def test_with_a_tag_it_returns_unused_images_carrying_it(self, db):
        await self._seed(db)
        resp = await _get(db, self.KEYS, tags="Kelly", no_video="true")
        assert resp.status_code == 200
        body = resp.json()
        assert [i["filename"] for i in body["items"]] == ["kelly.png"]
        assert body["total"] == 1

    async def test_alone_it_finds_images_with_no_meta_row(self, db):
        # A never-tagged image has no row; the scan is what makes it visible.
        await self._seed(db)
        resp = await _get(db, self.KEYS, no_video="true")
        assert resp.status_code == 200
        body = resp.json()
        assert [i["filename"] for i in body["items"]] == [
            "bare.png", "kelly.png",  # newest first, datasets hidden
        ]
        assert body["total"] == 2
        assert body["items"][1]["tags"] == "Kelly"

    async def test_alone_it_paginates(self, db):
        await self._seed(db)
        body = (await _get(db, self.KEYS, no_video="true", limit=1, offset=1)).json()
        assert [i["filename"] for i in body["items"]] == ["kelly.png"]
        assert body["total"] == 2

    async def test_without_it_an_empty_search_is_still_refused(self, db):
        resp = await _get(db, self.KEYS)
        assert resp.status_code == 400
