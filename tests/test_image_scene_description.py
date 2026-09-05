"""An image's scene description is produced once and kept (console#414).

JoyCaption costs 2070 time — 4.5s cold, 1.2s warm, on the box that also serves
Automatic1111 — and the same image starts many jobs. POST /captions/describe is
deliberately stateless, so until now every job paid for the same frame again.

The value is NOT a cache that can be rebuilt. The model is nondeterministic: describing the
frame again gives different words, and the words are what the person read and accepted. That
is why re-describing is an explicit action, and why nothing may throw the row away as a side
effect of an unrelated edit.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import sqlalchemy as sa

from app.models import ImageMeta

BUCKET = "wanly-images"
PATH = f"s3://{BUCKET}/2026-09-05/00001.png"
OTHER = f"s3://{BUCKET}/2026-09-05/00002.png"


class TestRowLifetime:
    """The row outlives the tags on it."""

    def test_a_row_holding_only_a_description_is_not_empty(self):
        meta = ImageMeta(path=PATH, tags=None, scene_description="a woman on a sofa")
        assert not meta.is_empty()

    def test_a_row_holding_only_tags_is_not_empty(self):
        assert not ImageMeta(path=PATH, tags="Kelly").is_empty()

    def test_a_row_holding_neither_is_empty(self):
        assert ImageMeta(path=PATH, tags="   ", scene_description="").is_empty()

    @pytest.mark.asyncio
    async def test_clearing_tags_keeps_the_description(self, db):
        """The tags endpoint used to delete the row outright. That threw away GPU work as a
        side effect of removing a tag."""
        db.add(ImageMeta(path=PATH, tags="Kelly", scene_description="a woman on a sofa",
                         scene_described_at=datetime.now(timezone.utc)))
        await db.flush()

        resp = await _patch_tags(db, PATH, None)
        assert resp.status_code == 200

        meta = await db.get(ImageMeta, PATH)
        assert meta is not None, "the row was deleted and the description with it"
        assert meta.tags is None
        assert meta.scene_description == "a woman on a sofa"

    @pytest.mark.asyncio
    async def test_clearing_tags_on_a_bare_row_still_removes_it(self, db):
        """`list_untagged_images` reads "no row" as "never tagged", so a row that says
        nothing must not survive and make an untagged image look handled."""
        db.add(ImageMeta(path=PATH, tags="Kelly"))
        await db.flush()

        await _patch_tags(db, PATH, None)
        assert await db.get(ImageMeta, PATH) is None


class TestDescribeEndpoint:
    @pytest.mark.asyncio
    async def test_it_stores_what_the_captioner_said(self, db):
        resp = await _post_scene(db, PATH, caption="a woman in a red dress on a sofa")
        assert resp.status_code == 200
        body = resp.json()
        assert body["scene_description"] == "a woman in a red dress on a sofa"
        assert body["words"] == 9
        assert body["scene_described_at"] is not None

        meta = await db.get(ImageMeta, PATH)
        assert meta.scene_description == "a woman in a red dress on a sofa"
        # HOW it was described, not only what was said.
        assert meta.scene_instruction

    @pytest.mark.asyncio
    async def test_describing_again_replaces_the_previous(self, db):
        """This endpoint IS the re-roll. It always regenerates — an "only if missing" rule
        here would make re-roll impossible to express."""
        await _post_scene(db, PATH, caption="first words")
        await _post_scene(db, PATH, caption="second words")

        meta = await db.get(ImageMeta, PATH)
        assert meta.scene_description == "second words"

    @pytest.mark.asyncio
    async def test_describing_does_not_disturb_the_tags(self, db):
        db.add(ImageMeta(path=PATH, tags="Kelly, Sofa"))
        await db.flush()

        await _post_scene(db, PATH, caption="a woman on a sofa")

        meta = await db.get(ImageMeta, PATH)
        assert meta.tags == "Kelly, Sofa"
        assert meta.scene_description == "a woman on a sofa"

    @pytest.mark.asyncio
    async def test_a_blank_caption_is_a_failure_not_a_description(self, db):
        """Storing "" would mark the image described and stop anything ever asking again."""
        resp = await _post_scene(db, PATH, caption="   ")
        assert resp.status_code == 503
        assert await db.get(ImageMeta, PATH) is None

    @pytest.mark.asyncio
    async def test_a_captioner_that_is_down_is_a_503(self, db):
        from app.joycaption import CaptionError

        resp = await _post_scene(db, PATH, error=CaptionError("2070 unreachable"))
        assert resp.status_code == 503
        assert await db.get(ImageMeta, PATH) is None

    @pytest.mark.asyncio
    async def test_a_path_outside_the_images_bucket_is_refused(self, db):
        resp = await _post_scene(db, "s3://wanly-jobs/x/seg0-last.png", caption="nope")
        assert resp.status_code == 400


class TestGetEndpoint:
    @pytest.mark.asyncio
    async def test_never_described_returns_nulls_not_404(self, db):
        """"No description yet" is an ordinary state of a perfectly good image, and the New
        Job modal has to render it either way."""
        resp = await _get_scene(db, PATH)
        assert resp.status_code == 200
        assert resp.json()["scene_description"] is None
        assert resp.json()["words"] == 0

    @pytest.mark.asyncio
    async def test_it_returns_what_was_stored(self, db):
        db.add(ImageMeta(path=PATH, scene_description="a woman on a sofa",
                         scene_instruction="describe it", 
                         scene_described_at=datetime.now(timezone.utc)))
        await db.flush()

        body = (await _get_scene(db, PATH)).json()
        assert body["scene_description"] == "a woman on a sofa"
        assert body["scene_instruction"] == "describe it"
        assert body["words"] == 5


class TestMoveCarriesTheRow:
    """A move used to drop the row, and with it the tags. Survivable for a tag; not for a
    description that cost GPU time and cannot be reproduced word for word."""

    @pytest.mark.asyncio
    async def test_the_description_follows_the_object(self, db):
        db.add(ImageMeta(path=PATH, tags="Kelly", scene_description="a woman on a sofa"))
        await db.flush()

        resp = await _move(db, ["2026-09-05/00001.png"], "2026-09-06")
        assert resp.status_code == 200

        assert await db.get(ImageMeta, PATH) is None
        moved = await db.get(ImageMeta, f"s3://{BUCKET}/2026-09-06/00001.png")
        assert moved is not None
        assert moved.tags == "Kelly"
        assert moved.scene_description == "a woman on a sofa"

    @pytest.mark.asyncio
    async def test_an_image_with_no_row_moves_without_inventing_one(self, db):
        resp = await _move(db, ["2026-09-05/00001.png"], "2026-09-06")
        assert resp.status_code == 200
        count = (await db.execute(sa.select(sa.func.count()).select_from(ImageMeta))).scalar()
        assert count == 0


# ---------------------------------------------------------------------------------------
# Harness. The routes are exercised through the app so the response models are covered too,
# with S3 and the captioner patched out — neither is reachable from a test run.
# ---------------------------------------------------------------------------------------

async def _client(db):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), app


async def _patch_tags(db, path, tags):
    client, app = await _client(db)
    try:
        async with client as c:
            return await c.patch("/images/tags", params={"path": path}, json={"tags": tags})
    finally:
        app.dependency_overrides.clear()


async def _get_scene(db, path):
    client, app = await _client(db)
    try:
        async with client as c:
            return await c.get("/images/scene", params={"path": path})
    finally:
        app.dependency_overrides.clear()


async def _post_scene(db, path, caption=None, error=None):
    client, app = await _client(db)
    describe = (
        patch("app.routes.images.caption_image_bytes", side_effect=error) if error
        else patch("app.routes.images.caption_image_bytes",
                   return_value=(caption, "an instruction"))
    )
    try:
        with patch("app.routes.images.download_bytes", return_value=b"png"), describe:
            async with client as c:
                return await c.post("/images/scene", params={"path": path}, json={})
    finally:
        app.dependency_overrides.clear()


async def _move(db, keys, target):
    client, app = await _client(db)
    try:
        with patch("app.routes.images.move_object"):
            async with client as c:
                return await c.post("/images/move",
                                    json={"keys": keys, "target_folder": target})
    finally:
        app.dependency_overrides.clear()
