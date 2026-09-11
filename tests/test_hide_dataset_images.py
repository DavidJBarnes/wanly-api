"""Dataset images never appear in the Image Repo (wanly-api#309).

Search runs under repo_images_only and the folder listing hides the datasets prefix, both
from wanly-console#464's split. Two views still leaked: /images/untagged scanned every
common prefix, and /images/favorites rendered whatever URIs were favorited. A dataset
staging image entering either grid made the Image Repo look connected to the datasets —
which is exactly the look the split was for.
"""

from unittest.mock import patch

import pytest

BUCKET = "wanly-images"
REPO = f"s3://{BUCKET}/2026-09-05/00001.png"
DATASET = f"s3://{BUCKET}/datasets/faces-abc/staging.png"
DATASET2 = f"s3://{BUCKET}/datasets/faces-abc/other.png"

def _obj(key, size=1024, modified="2026-09-05T00:00:00Z"):
    return {"Key": key, "Size": size, "LastModified": modified}


class _FakeS3:
    """list_objects / list_common_prefixes / head_object, answered from a fixed key set."""

    def __init__(self, keys):
        self._keys = keys

    def list_common_prefixes(self, bucket):
        markers = []
        for k in self._keys:
            rest = k[len(f"s3://{bucket}/"):]
            top = rest.split("/", 1)[0]
            if top not in markers:
                markers.append(top)
        return [m for m in markers if m != "datasets"]

    def list_objects(self, bucket, prefix):
        return [_obj(k[len(f"s3://{bucket}/"):]) for k in self._keys
                if k.startswith(f"s3://{bucket}/{prefix}") and not k.endswith("/.folder")]

    def head_object(self, uri):
        key = uri.split("/", 3)[-1] if uri.count("/") > 2 else uri
        return {"Key": key, "Size": 1024, "LastModified": "2026-09-05T00:00:00Z"} \
            if f"s3://{BUCKET}/{key}" in self._keys else None


async def _client(db):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), app


@pytest.fixture
def s3():
    fake = _FakeS3({REPO, DATASET, DATASET2})
    with patch("app.routes.images.list_common_prefixes", fake.list_common_prefixes), \
         patch("app.routes.images.list_objects", fake.list_objects), \
         patch("app.routes.images.head_object", fake.head_object):
        yield fake


@pytest.mark.asyncio
class TestUntaggedHidesDatasets:
    async def test_a_dataset_staging_image_never_appears_as_untagged(self, db, s3):
        """The untagged view scanned every common prefix; a dataset image with no tags
        entered the grid and made the Image Repo look connected to the datasets."""
        client, app = await _client(db)
        try:
            async with client as c:
                resp = await c.get("/images/untagged")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        paths = {i["path"] for i in resp.json()}
        assert DATASET not in paths and DATASET2 not in paths
        assert REPO in paths, "the view must keep showing repo images"

    async def test_the_dataset_folder_is_not_even_scanned(self, db, s3):
        """Prefix scanning walks what list_common_prefixes returns; the check is at the
        object level so it holds regardless of what prefixes arrive."""
        keys = {f"s3://{BUCKET}/datasets/x.png", REPO}
        fake = _FakeS3(keys)

        def prefixes(bucket):
            return [f"{DATASET.split('/', 4)[4].split('/')[0]}"] if False else ["datasets/", "2026-09-05/"]

        with patch("app.routes.images.list_common_prefixes", prefixes), \
             patch("app.routes.images.list_objects", fake.list_objects), \
             patch("app.routes.images.head_object", fake.head_object):
            client, app = await _client(db)
            try:
                async with client as c:
                    resp = await c.get("/images/untagged")
            finally:
                app.dependency_overrides.clear()
        paths = {i["path"] for i in resp.json()}
        assert f"s3://{BUCKET}/datasets/x.png" not in paths


@pytest.mark.asyncio
class TestFavoritesHidesDatasets:
    async def test_a_favorited_dataset_image_is_not_rendered(self, db, s3):
        """A stale favorite on a dataset image drops out of the response rather than
        surfacing as a grid entry; the Favorite row itself stays."""
        from app.models import Favorite, User
        from app.auth import get_current_user
        import uuid

        user = User(id=uuid.uuid4(), username="uf", password_hash="x")
        db.add(user)
        await db.flush()
        me = type("Me", (), {"id": user.id})()
        db.add(Favorite(user_id=me.id, item_type="image", item_ref=DATASET))
        db.add(Favorite(user_id=me.id, item_type="image", item_ref=REPO))
        await db.flush()

        client, app = await _client(db)
        app.dependency_overrides[get_current_user] = lambda: me
        try:
            async with client as c:
                resp = await c.get("/images/favorites")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        paths = {i["path"] for i in resp.json()}
        assert DATASET not in paths
        assert REPO in paths, "the favorites view must keep showing repo images"

    async def test_the_favorite_row_itself_survives(self, db, s3):
        """This endpoint renders, it does not curate. Un-hiding must not lose data — the
        row stays so a deliberate dataset favorite still exists underneath."""
        import uuid
        from app.models import Favorite, User
        from app.auth import get_current_user

        user = User(id=uuid.uuid4(), username="ur", password_hash="x")
        db.add(user)
        await db.flush()
        me = type("Me", (), {"id": user.id})()
        db.add(Favorite(user_id=me.id, item_type="image", item_ref=DATASET))
        await db.flush()

        client, app = await _client(db)
        app.dependency_overrides[get_current_user] = lambda: me
        try:
            async with client as c:
                resp = await c.get("/images/favorites")
        finally:
            app.dependency_overrides.clear()
        assert resp.status_code == 200
        assert resp.json() == []
        # The row is still there.
        from sqlalchemy import select
        rows = (await db.execute(select(Favorite))).scalars().all()
        assert any(f.item_ref == DATASET for f in rows)
