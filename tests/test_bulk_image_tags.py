"""Bulk add/remove tags across a selection (console#517).

The per-image PATCH replaces the whole blob with what the browser last saw, so a client
loop over it would race the lightbox and duplicate tags the search clause's normalisation
already hides ("Kelly" beside "kelly "). The merge therefore happens server-side, in one
transaction, deduped through tag_filter.normalise_tag.
"""

from unittest.mock import patch  # noqa: F401  (kept for parity with sibling tests)

import pytest
import pytest_asyncio

from app.models import ImageMeta

BUCKET = "wanly-images"
A = f"s3://{BUCKET}/2026-09-17/00001.png"
B = f"s3://{BUCKET}/2026-09-17/00002.png"
C = f"s3://{BUCKET}/2026-09-17/00003.png"


@pytest_asyncio.fixture(autouse=True)
def _no_real_describe_task(monkeypatch):
    """Stub the background entry point for every test in this module.

    Starlette runs BackgroundTasks inside the ASGI call, so without this every tagging
    test would open a real session and reach for S3 and the captioner after its response.
    Tests that care about the queue patch over this with their own fake.
    """
    from app.routes import images

    async def noop(paths):
        pass

    monkeypatch.setattr(images, "describe_untagged_job", noop)


class TestBulkAdd:
    @pytest.mark.asyncio
    async def test_it_tags_every_selected_image(self, db):
        resp = await _bulk(db, [A, B], "Kelly, Missionary")
        assert resp.status_code == 200
        assert (await db.get(ImageMeta, A)).tags == "Kelly, Missionary"
        assert (await db.get(ImageMeta, B)).tags == "Kelly, Missionary"

    @pytest.mark.asyncio
    async def test_it_merges_into_existing_tags(self, db):
        db.add(ImageMeta(path=A, tags="Sofa"))
        await db.flush()

        await _bulk(db, [A], "Kelly")
        assert (await db.get(ImageMeta, A)).tags == "Sofa, Kelly"

    @pytest.mark.asyncio
    async def test_a_normalised_duplicate_is_not_added_again(self, db):
        """The case the client-side loop could not get right: 'kelly ' is already Kelly.
        The merge rewrites the blob canonically ("kelly, Sofa"), so assert on the tag SET
        — what counts is that Kelly appears once."""
        db.add(ImageMeta(path=A, tags="kelly , Sofa"))
        await db.flush()

        await _bulk(db, [A], "Kelly, sofa")
        meta = await db.get(ImageMeta, A)
        assert sorted(meta.tags.lower().split(", ")) == ["kelly", "sofa"]

    @pytest.mark.asyncio
    async def test_the_incoming_set_dedupes_itself(self, db):
        await _bulk(db, [A], "Kelly, kelly, Kelly ")
        assert (await db.get(ImageMeta, A)).tags == "Kelly"

    @pytest.mark.asyncio
    async def test_a_description_survives_the_add(self, db):
        db.add(ImageMeta(path=A, tags="Sofa", scene_description="a woman on a sofa"))
        await db.flush()

        await _bulk(db, [A], "Kelly")
        meta = await db.get(ImageMeta, A)
        assert meta.tags == "Sofa, Kelly"
        assert meta.scene_description == "a woman on a sofa"


class TestBulkRemove:
    @pytest.mark.asyncio
    async def test_it_drops_only_whole_tag_matches(self, db):
        """Removing Kelly must not touch KellyTeacher — the substring trap tag_filter
        exists to stop."""
        db.add(ImageMeta(path=A, tags="Kelly, KellyTeacher"))
        db.add(ImageMeta(path=B, tags="kellyyoung"))
        await db.flush()

        await _bulk(db, [A, B], "Kelly", mode="remove")
        assert (await db.get(ImageMeta, A)).tags == "KellyTeacher"
        assert (await db.get(ImageMeta, B)).tags == "kellyyoung"

    @pytest.mark.asyncio
    async def test_emptied_bare_row_is_deleted(self, db):
        """No row means "never tagged" to the untagged view; a row that says nothing must
        not survive."""
        db.add(ImageMeta(path=A, tags="Kelly"))
        await db.flush()

        await _bulk(db, [A], "Kelly", mode="remove")
        assert await db.get(ImageMeta, A) is None

    @pytest.mark.asyncio
    async def test_emptied_row_with_a_description_is_kept(self, db):
        """console#414: the row carries GPU-produced captions; emptying its tags only
        deletes it when is_empty() agrees."""
        db.add(ImageMeta(path=A, tags="Kelly", scene_description="a woman on a sofa"))
        await db.flush()

        await _bulk(db, [A], "Kelly", mode="remove")
        meta = await db.get(ImageMeta, A)
        assert meta is not None
        assert meta.tags is None
        assert meta.scene_description == "a woman on a sofa"

    @pytest.mark.asyncio
    async def test_removing_from_an_image_with_no_row_invents_nothing(self, db):
        await _bulk(db, [A], "Kelly", mode="remove")
        assert await db.get(ImageMeta, A) is None


class TestGuards:
    @pytest.mark.asyncio
    async def test_a_path_outside_the_images_bucket_refuses_the_request(self, db):
        resp = await _bulk(db, [A, "s3://wanly-jobs/j1/x.png"], "Kelly")
        assert resp.status_code == 400
        assert await db.get(ImageMeta, A) is None

    @pytest.mark.asyncio
    async def test_an_overflowing_merge_refuses_everything(self, db):
        """Half-tagging a selection is worse than refusing it: the user cannot tell which
        images got the tag."""
        filler = ", ".join(f"tag{i:04d}" for i in range(55))  # 493 chars, under the cap
        db.add(ImageMeta(path=A, tags=filler))
        db.add(ImageMeta(path=B, tags="Sofa"))
        await db.flush()

        resp = await _bulk(db, [A, B], "a-tag-nobody-would-ever-add")
        assert resp.status_code == 400
        assert A in resp.json()["detail"]["paths"]
        assert (await db.get(ImageMeta, A)).tags == filler
        assert (await db.get(ImageMeta, B)).tags == "Sofa", "B was written despite the refusal"

    @pytest.mark.asyncio
    async def test_empty_tag_string_is_refused(self, db):
        resp = await _bulk(db, [A], " , , ")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_the_response_reports_each_path_so_caches_can_patch(self, db):
        db.add(ImageMeta(path=A, tags="Sofa"))
        await db.flush()

        resp = await _bulk(db, [A, B], "Sofa")
        body = resp.json()
        by_path = {r["path"]: r for r in body["results"]}
        assert by_path[A]["tags"] == "Sofa"
        assert by_path[A]["changed"] is False
        assert by_path[B]["tags"] == "Sofa"
        assert by_path[B]["changed"] is True


# ---------------------------------------------------------------------------------------
# Harness — same shape as test_image_scene_description.py.
# ---------------------------------------------------------------------------------------

async def _bulk(db, paths, tags, mode="add"):
    from httpx import ASGITransport, AsyncClient
    from app.auth import get_current_user
    from app.database import get_db
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_db] = lambda: db
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as c:
            return await c.post("/images/tags",
                                json={"paths": paths, "tags": tags, "mode": mode})
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------------------
# Auto-describe (api#340): tagging is the moment an image becomes worth keeping — the
# rule console#414 built single-image auto-describe on — so a bulk add that newly tags a
# never-described image queues its description server-side, serially, after the commit.
# ---------------------------------------------------------------------------------------

class TestAutoDescribeQueueing:
    """What the endpoint puts on the background queue."""

    @pytest.mark.asyncio
    async def test_an_add_that_newly_tags_reports_the_describe_count(self, db):
        resp = await _bulk(db, [A, B], "Kelly")
        assert resp.json()["describing"] == 2

    @pytest.mark.asyncio
    async def test_an_already_described_image_is_not_queued_again(self, db):
        db.add(ImageMeta(path=A, tags="Sofa", scene_description="a woman on a sofa"))
        await db.flush()

        resp = await _bulk(db, [A, B], "Kelly")
        assert resp.json()["describing"] == 1

    @pytest.mark.asyncio
    async def test_an_add_that_changes_nothing_describes_nothing(self, db):
        db.add(ImageMeta(path=A, tags="Kelly"))
        await db.flush()

        resp = await _bulk(db, [A], "Kelly")
        assert resp.json()["describing"] == 0

    @pytest.mark.asyncio
    async def test_a_remove_never_describes(self, db):
        db.add(ImageMeta(path=A, tags="Kelly, Sofa"))
        await db.flush()

        resp = await _bulk(db, [A], "Kelly", mode="remove")
        assert resp.json()["describing"] == 0

    @pytest.mark.asyncio
    async def test_the_queued_paths_are_exactly_the_undescribed_ones(self, db, monkeypatch):
        """Empty-string description counts as undescribed: storing "" is refused at the
        describe endpoint, so a blank must not silence the batch."""
        from app.routes import images

        db.add(ImageMeta(path=A, tags="Sofa", scene_description="a woman on a sofa"))
        db.add(ImageMeta(path=B, tags="Sofa", scene_description="   "))
        await db.flush()

        queued = []

        async def fake_job(paths):
            queued.extend(paths)

        monkeypatch.setattr(images, "describe_untagged_job", fake_job)

        await _bulk(db, [A, B, C], "Kelly")
        assert queued == [B, C]


class TestDescribeUntagged:
    """The loop itself: serial, provenance-complete, and it stops at the first refusal."""

    def _patch_captioner(self, monkeypatch, *, caption="a woman on a sofa",
                         motion="she leans back", fail_with=None):
        """Fake the download + caption pair; returns the list of caption calls."""
        from app.routes import images as images_mod
        from app.routes.captions import ScenePair

        called = []

        def fake_download(path):  # sync, like boto3: the loop runs it in a thread
            return b"png bytes"

        async def fake_pair(db, image, **kw):
            called.append(image)
            if fail_with is not None:
                raise fail_with
            return ScenePair(scene=caption, scene_instruction="the instruction",
                             motion=motion, motion_instruction="the motion instruction")

        monkeypatch.setattr(images_mod, "download_bytes", fake_download)
        monkeypatch.setattr(images_mod, "caption_image_pair", fake_pair)
        return called

    @pytest.mark.asyncio
    async def test_it_describes_a_newly_tagged_row_completely(self, db, monkeypatch):
        from app.routes.images import describe_untagged

        db.add(ImageMeta(path=A, tags="Kelly"))
        await db.flush()
        self._patch_captioner(monkeypatch)

        assert await describe_untagged(db, [A]) == 1
        meta = await db.get(ImageMeta, A)
        assert meta.scene_description == "a woman on a sofa"
        assert meta.scene_instruction == "the instruction"
        assert meta.scene_described_at is not None
        # Same call writes both halves, so the provenance matches POST /images/scene's.
        assert meta.motion_description == "she leans back"
        assert meta.motion_described_at is not None
        # And it did not disturb the tags that queued it.
        assert meta.tags == "Kelly"

    @pytest.mark.asyncio
    async def test_an_already_described_row_is_left_alone(self, db, monkeypatch):
        """The queue is computed at commit time; a re-run must not re-roll a caption.
        GPU work the user already accepted is not ours to replace."""
        from app.routes.images import describe_untagged

        db.add(ImageMeta(path=A, tags="Kelly", scene_description="a woman on a sofa"))
        await db.flush()
        called = self._patch_captioner(monkeypatch)

        assert await describe_untagged(db, [A]) == 0
        assert called == []
        assert (await db.get(ImageMeta, A)).scene_description == "a woman on a sofa"

    @pytest.mark.asyncio
    async def test_a_busy_captioner_stops_the_batch_without_touching_anything(self, db,
                                                                              monkeypatch):
        from app.joycaption import CaptionerBusy
        from app.routes.images import describe_untagged

        db.add(ImageMeta(path=A, tags="Kelly"))
        db.add(ImageMeta(path=B, tags="Kelly"))
        await db.flush()
        called = self._patch_captioner(
            monkeypatch, fail_with=CaptionerBusy("3090 is rendering"))

        assert await describe_untagged(db, [A, B]) == 0
        assert len(called) == 1, "the batch must stop, not hammer a busy box"
        assert (await db.get(ImageMeta, A)).scene_description is None
        assert (await db.get(ImageMeta, B)).scene_description is None

    @pytest.mark.asyncio
    async def test_an_unreachable_captioner_also_stops_the_batch(self, db, monkeypatch):
        from app.joycaption import CaptionError
        from app.routes.images import describe_untagged

        db.add(ImageMeta(path=A, tags="Kelly"))
        await db.flush()
        self._patch_captioner(monkeypatch, fail_with=CaptionError("captioner unreachable"))

        assert await describe_untagged(db, [A]) == 0

    @pytest.mark.asyncio
    async def test_an_empty_caption_is_not_stored(self, db, monkeypatch):
        """Storing "" marks the image described and stops anything ever asking again."""
        from app.routes.images import describe_untagged

        db.add(ImageMeta(path=A, tags="Kelly"))
        await db.flush()
        self._patch_captioner(monkeypatch, caption="   ")

        assert await describe_untagged(db, [A]) == 0
        assert (await db.get(ImageMeta, A)).scene_description is None

    @pytest.mark.asyncio
    async def test_one_unreadable_image_does_not_stop_the_rest(self, db, monkeypatch):
        """A deleted file is that image's problem; the captioner is fine."""
        from app.routes import images as images_mod
        from app.routes.images import describe_untagged

        db.add(ImageMeta(path=A, tags="Kelly"))
        db.add(ImageMeta(path=B, tags="Kelly"))
        await db.flush()
        self._patch_captioner(monkeypatch)

        def flaky_download(path):
            if path == A:
                raise FileNotFoundError("gone from s3")
            return b"png bytes"

        monkeypatch.setattr(images_mod, "download_bytes", flaky_download)

        assert await describe_untagged(db, [A, B]) == 1
        assert (await db.get(ImageMeta, B)).scene_description == "a woman on a sofa"

    @pytest.mark.asyncio
    async def test_a_row_deleted_after_the_commit_is_skipped_not_created(self, db,
                                                                         monkeypatch):
        """Tagged, then deleted between commit and the background task."""
        from app.routes.images import describe_untagged

        called = self._patch_captioner(monkeypatch)
        assert await describe_untagged(db, [A]) == 0
        assert called == []
        assert await db.get(ImageMeta, A) is None
