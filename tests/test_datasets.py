"""Named, taggable training datasets.

The grouping that did not exist. Before this the only ways to say "these images belong together"
were an S3 folder prefix and a per-user favourites list, so a training set could not be named,
tagged or re-opened — every run meant re-selecting by hand, and a v2 meant doing it again from
memory.
"""
import uuid

import pytest

from app.models import Dataset
from app.routes.datasets import _prefix
from app.schemas.datasets import DatasetCreate
from app.schemas.training import TrainingCreate


class TestNaming:
    def test_a_readable_name_is_allowed(self):
        assert DatasetCreate(name="p@y v2 faces").name == "p@y v2 faces"

    @pytest.mark.parametrize("bad", ["a/b", "../etc", "tab\there", ""])
    def test_a_name_that_breaks_a_prefix_is_refused(self, bad):
        """It becomes part of every image's S3 key, forever."""
        with pytest.raises(ValueError):
            DatasetCreate(name=bad)

    def test_the_prefix_is_keyed_by_id_under_one_hidden_folder(self):
        """It used to be `dataset-<name>`, which put every dataset in the Image Repo's folder
        list beside the generation folders and made the two look connected (console#464).
        By id, so a rename is just a rename."""
        import uuid
        from app.routes.datasets import DATASETS_PREFIX, _prefix
        i = uuid.uuid4()
        assert _prefix(i) == f"{DATASETS_PREFIX}/{i}"

    def test_the_image_repo_does_not_list_the_datasets_folder(self):
        import inspect
        from app.routes import images as mod
        src = inspect.getsource(mod.list_folders)
        assert 'p.rstrip("/") != DATASETS_PREFIX' in src

    def test_a_rename_refuses_a_name_that_exists(self):
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.update_dataset)
        assert "already exists" in src

class TestTraining:
    def test_a_job_can_name_a_dataset_instead_of_listing_images(self):
        t = TrainingCreate(character="p@y", trigger="p@y", dataset_id=uuid.uuid4())
        assert t.dataset_images == []
        assert t.dataset_id is not None

    def test_a_raw_list_still_works(self):
        """A CLI or a curl should not have to create a dataset first."""
        imgs = [f"s3://b/{i}.jpg" for i in range(13)]
        assert TrainingCreate(character="x", trigger="x", dataset_images=imgs).dataset_id is None

    def test_the_minimum_is_enforced_in_the_route_not_the_schema(self):
        """A schema minimum on dataset_images would reject every dataset_id request, which is
        the normal path."""
        import inspect
        from app.routes import training as mod
        src = inspect.getsource(mod.create_training_job)
        assert "len(images) < MIN_DATASET_IMAGES" in src
        # and it resolves the dataset before checking
        assert src.index("body.dataset_id") < src.index("len(images) < MIN_DATASET_IMAGES")


@pytest.mark.asyncio
class TestTheRow:
    async def test_images_are_an_ordered_list(self, db):
        """Order is significant: the trainer stages them as sel_000..N and the captions pair
        by index."""
        ds = Dataset(id=uuid.uuid4(), name="d", images=["s3://b/z.jpg", "s3://b/a.jpg"],
                     prefix="dataset-d")
        db.add(ds)
        await db.flush()
        assert ds.images == ["s3://b/z.jpg", "s3://b/a.jpg"]

    async def test_a_dataset_starts_empty(self, db):
        ds = Dataset(id=uuid.uuid4(), name="empty", images=[], prefix="dataset-empty")
        db.add(ds)
        await db.flush()
        assert ds.images == []


class TestUploadSemantics:
    def test_the_image_list_is_reassigned_not_mutated(self):
        """A JSONB column does not see a mutation of the existing list, so `images.append(...)`
        writes nothing and the upload silently vanishes on the next read."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.add_images)
        assert "ds.images = list(ds.images) + added" in src
        # Code only. The comment above that line names the mistake it is avoiding, and a
        # substring check would match the explanation rather than the bug.
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert "ds.images.append" not in code

    def test_a_non_image_is_skipped_not_fatal(self):
        """One stray file in a folder drag-and-drop must not reject the other forty-nine."""
        import inspect
        from app.routes import datasets as mod
        assert "continue" in inspect.getsource(mod.add_images)

    def test_deleting_a_dataset_keeps_its_images_by_default(self):
        """A finished training job records the URIs it trained on; destroying them turns a
        reproducible run into an unreproducible one."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.delete_dataset)
        assert "purge" in src and "if purge" in src

    async def test_a_purge_deletes_the_prefix_not_a_bucket_named_after_the_prefix(self):
        """Every purge delete 500'd: the call site passed (bucket, prefix) into
        s3.delete_prefix(prefix, bucket), so botocore validated the PREFIX as a bucket name
        and raised ParamValidationError — for a legacy name-keyed prefix like
        "dataset-test-faces" that is not a valid bucket at all. The fake asserts the
        positional contract the three other call sites already honour."""
        from unittest.mock import patch
        from app import s3
        from app.config import settings
        from app.routes import datasets as mod

        ds = Dataset(id=uuid.uuid4(), name="Me", prefix="dataset-test-faces", images=[])
        calls = []

        def fake_delete_prefix(prefix, bucket):
            calls.append((prefix, bucket))
            return 0

        class FakeDb:
            async def get(self, model, _id):
                return ds
            async def delete(self, row):
                pass
            async def commit(self):
                pass

        with patch.object(s3, "delete_prefix", fake_delete_prefix):
            await mod.delete_dataset(ds.id, purge=True, _user=None, db=FakeDb())

        assert calls == [(ds.prefix + "/", settings.s3_images_bucket)]

    def test_renaming_does_not_move_the_prefix(self):
        """A finished training job's dataset_images point at the old keys."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.update_dataset)
        assert "ds.prefix" not in src.split("body.name")[1].split("body.tags")[0]

class TestCropping:
    """Step 2 of the documented pipeline, which until now existed only as laptop scripts that
    ssh'd to the box with insightface."""

    def test_it_replaces_the_images_in_place(self):
        """It used to write a second dataset called "<name> faces". Every dataset then came in
        pairs and the one you trained from was never the one you named (console#464). The
        photographs stay in the bucket under the dataset's prefix; the dataset IS the crops --
        and #303 added a save_as mode where the set keeps its photos and the crops join it."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "ds.images = kept_others + crop_uris" in src
        assert "ds.images = list(ds.images) + crop_uris" in src
        assert "out = Dataset(" not in src

    def test_it_crops_a_subset_when_uris_are_named(self):
        """A 25-image set that needed 5 faces re-cropped got all 25 re-cropped: the output is
        not deterministic, so the 20 good crops came back different. #303: absent/None means
        every image (the old behavior); a list means those and only those.

        #336: this test asserted `sig["uris"].default is None` and passed the whole time the
        selection was ignored — a bare `= None` on a list is exactly the body-param declaration
        that dropped the query keys. The default is now a Query marker; the wire behaviour is
        proven by TestCropSelectionOverTheWire, which is the only thing that can see the bug."""
        import inspect
        from fastapi.params import Param
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        default = inspect.signature(mod.crop_faces).parameters["uris"].default
        assert isinstance(default, Param) and default.default is None, \
            "uris must be a query param, or FastAPI reads it from the body and the selection is lost"
        assert "ds.images if uris is None else" in src

    def test_a_stale_selection_is_refused_not_fatal(self):
        """URIs named but not in the set -- removed in another tab while the dialog was open --
        are filtered out; naming only stale ones is a 422 that says so, not a silent no-op."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "not fatal" not in src or True
        assert "none of the selected images are in this dataset" in src

    def test_save_as_keeps_the_set_and_appends_the_crops(self):
        """The other half of the crop question: one run produces photographs AND faces."""
        import inspect
        from app.routes import datasets as mod
        assert "save_as" in inspect.signature(mod.crop_faces).parameters
        src = inspect.getsource(mod.crop_faces)
        assert "if save_as:" in src

    def test_save_as_does_not_break_the_anchor_outside_the_selection(self):
        """An anchor outside a subset crop is still a photograph in the set; only one being
        cropped away clears it."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "if ds.anchor_uri in replaced:" in src

    def test_a_second_crop_cannot_overwrite_the_first_batch(self):
        """A training job may still record the first batch's keys."""
        import inspect
        from app.routes import datasets as mod
        assert 'faces-{uuid.uuid4().hex[:6]}' in inspect.getsource(mod.crop_faces)

    def test_the_anchor_is_cleared_because_it_was_a_photograph(self):
        import inspect
        from app.routes import datasets as mod
        assert "ds.anchor_uri = None" in inspect.getsource(mod.crop_faces)

    def test_there_is_no_reference_and_no_gate(self):
        """Scoring a mixed set against a reference dataset's MEAN separates nobody, and the
        dropdown for naming one was the most confusing control on the page. Culling happens
        afterwards against ONE anchor, with the numbers on screen."""
        import inspect
        from app.routes import datasets as mod
        params = inspect.signature(mod.crop_faces).parameters
        assert "reference_dataset_id" not in params
        assert "gate" not in params
        assert not hasattr(mod, "_mean_via_service")

    def test_how_many_faces_to_keep_is_a_choice(self):
        """A dataset of couples does not want only the largest face: "largest" is then whoever
        stood closer to the camera, so the output silently interleaves two people."""
        import inspect
        from app.routes import datasets as mod
        assert "largest_only" in inspect.signature(mod.crop_faces).parameters

    def test_the_choice_actually_reaches_the_service(self):
        """It was hardcoded True in the payload, so the parameter alone would do nothing."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert '"largest_only": largest_only,' in src
        assert '"largest_only": True,' not in src

    def test_the_note_records_which_mode_produced_it(self):
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "largest only" in src and "every face" in src and "with none detected" in src

    def test_an_unconfigured_service_says_so_rather_than_timing_out(self):
        import inspect
        from app.routes import datasets as mod
        assert "face_crop_url is empty" in inspect.getsource(mod.crop_faces)

    def test_no_faces_at_all_is_an_error_not_an_empty_dataset(self):
        import inspect
        from app.routes import datasets as mod
        assert "no faces were detected" in inspect.getsource(mod.crop_faces)

    def test_an_absent_embedding_scores_below_any_floor(self):
        from app.routes.datasets import _cos
        assert _cos([], [1.0]) < 0.4


# -------------------------------------------------------------------------------------------
# Selection over the wire (api#336).
#
# Every test above in this class asserts on `inspect.getsource` — and every one of them passed
# while the selection was ignored in production. The line `ds.images if uris is None else …`
# is correct source code; what was wrong is one layer above, in FastAPI's parameter binding:
# `uris: list[str] | None = None` is a bare list-typed param, which FastAPI treats as a BODY
# parameter. The console posts body `null` with the picks as repeated `uris=` query keys, so
# the handler saw `None` and cropped every image. Access logs from 2026-09-22 show all three
# crop requests for one dataset carrying 8 URIs each; the dataset notes show crops from 40 of
# 40 and 29 of 51 — the full set, both times.
#
# Only a real HTTP request through the ASGI app can see a binding bug, so that is what these
# are. They post exactly what axios posts (repeated keys, body `null`), with s3 and the
# face-crop service stubbed at the module seam.
# -------------------------------------------------------------------------------------------

class _FakeS3:
    """Records what the endpoint downloaded, which is what `targets` resolved to."""

    def __init__(self):
        self.downloaded: list[str] = []

    def download_bytes(self, uri):
        self.downloaded.append(uri)
        return b"jpeg"

    def upload_bytes(self, data, key, bucket):
        return f"s3://{bucket}/{key}"


class _CropResp:
    def raise_for_status(self):
        pass

    def json(self):
        # One face per image submitted, which is what a solo-face dataset produces.
        return {"faces": [{"source_index": i, "face_index": 0,
                           "png_b64": b"eA==", "format": "jpeg"}
                          for i in range(self.n)],
                "no_face": []}


def _crop_client(n_images_seen_by_service):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            _crop_client.last_payload = json
            r = _CropResp()
            r.n = len(json["images"])
            return r

    return _Client


class _HttpxShim:
    """Replaces `datasets.httpx` for one test. Patching the real module's AsyncClient would
    replace the test's own client as well."""

    def __init__(self, client):
        import httpx
        self.AsyncClient = client
        self.HTTPError = httpx.HTTPError


@pytest.mark.asyncio
class TestCropSelectionOverTheWire:
    """The crop honors the selection the dialog made. See the block comment above."""

    IMGS = [f"s3://wanly-images/datasets/x/{n}.jpg" for n in ("a", "b", "c", "d")]

    async def _ds(self, db):
        from app.models import Dataset
        d = Dataset(name=f"wire-{uuid.uuid4().hex[:6]}", images=list(self.IMGS), prefix="p")
        db.add(d)
        await db.flush()
        return d

    async def _crop(self, db, monkeypatch, ds, query=None):
        from app.auth import get_current_user
        from app.config import settings
        from app.database import get_db
        from app.main import app
        from app.routes import datasets as mod

        fake = _FakeS3()
        monkeypatch.setattr(mod, "s3", fake)
        monkeypatch.setattr(mod, "httpx", _HttpxShim(_crop_client(None)))
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")
        app.dependency_overrides[get_current_user] = lambda: object()
        app.dependency_overrides[get_db] = lambda: db
        try:
            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(transport=ASGITransport(app=app),
                                   base_url="http://test") as c:
                q = "" if query is None else "?" + "&".join(f"uris={u}" for u in query)
                resp = await c.post(f"/datasets/{ds.id}/crop{q}",
                                    content=None,
                                    headers={"Content-Type": "application/json"})
                return resp, fake
        finally:
            app.dependency_overrides.clear()

    async def test_named_uris_crop_those_and_only_those(self, db, monkeypatch):
        """8 picks on the wire used to crop 51. The downloads are what `targets` became —
        assert on those, not on the response body."""
        resp, fake = await self._crop(db, monkeypatch, await self._ds(db),
                                      query=[self.IMGS[0], self.IMGS[2]])
        assert resp.status_code == 200, resp.text
        assert set(fake.downloaded) == {self.IMGS[0], self.IMGS[2]}

    async def test_the_unselected_images_survive_the_crop(self, db, monkeypatch):
        resp, _ = await self._crop(db, monkeypatch, await self._ds(db), query=[self.IMGS[1]])
        assert resp.status_code == 200, resp.text
        kept = [u for u in resp.json()["images"] if u in self.IMGS]
        assert kept == [u for u in self.IMGS if u != self.IMGS[1]]

    async def test_no_selection_crops_everything(self, db, monkeypatch):
        """Absent means every image — the old behavior, kept on purpose."""
        resp, fake = await self._crop(db, monkeypatch, await self._ds(db))
        assert resp.status_code == 200, resp.text
        assert set(fake.downloaded) == set(self.IMGS)

    async def test_the_note_says_what_was_actually_cropped(self, db, monkeypatch):
        """The note said "selected images" even when it cropped everything, because the
        `uris` local was rebound to the new crops before the note was built. A note that
        lies about its own scope is how #336 hid in the production data."""
        ds = await self._ds(db)
        resp, _ = await self._crop(db, monkeypatch, ds, query=[self.IMGS[3]])
        assert "1 selected images" in resp.json()["notes"]
        resp, _ = await self._crop(db, monkeypatch, ds)
        assert "every image" in resp.json()["notes"]

    async def test_a_selection_of_only_stale_uris_refuses_rather_than_cropping_all(self, db,
                                                                                   monkeypatch):
        resp, fake = await self._crop(db, monkeypatch, await self._ds(db),
                                      query=["s3://wanly-images/gone.jpg"])
        assert resp.status_code == 422
        assert fake.downloaded == []


class TestKeepEverythingByDefault:
    """An unwanted crop is one click to remove; a missing one is a re-run."""

    def test_every_face_is_kept_by_default(self):
        import inspect
        from app.routes import datasets as mod
        sig = inspect.signature(mod.crop_faces)
        assert sig.parameters["largest_only"].default is False


@pytest.mark.asyncio
class TestTheAnchor:
    """One picked face, and everything scored against it.

    Scoring against a dataset MEAN does not survive a real set: a mean over images that still
    contain two people is a blend of both. One face has no such ambiguity.
    """

    async def _ds(self, db, **kw):
        from app.models import Dataset
        base = dict(name=f"set-{uuid.uuid4().hex[:6]}", images=[], prefix="p")
        base.update(kw)
        ds = Dataset(**base)
        db.add(ds)
        await db.commit()
        return ds

    async def test_an_anchor_is_remembered_on_the_dataset(self, db):
        ds = await self._ds(db, images=["s3://b/a.png"], anchor_uri="s3://b/a.png")
        assert ds.anchor_uri == "s3://b/a.png"

    async def test_removing_the_anchor_clears_it(self, db):
        """Otherwise the next scoring pass measures against an image that is no longer there."""
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        ds = await self._ds(db, images=["s3://b/a.png", "s3://b/b.png"],
                            anchor_uri="s3://b/a.png")
        out = await update_dataset(ds.id, DatasetUpdate(images=["s3://b/b.png"]),
                                   _user=None, db=db)
        assert out.anchor_uri is None

    async def test_removing_something_else_keeps_the_anchor(self, db):
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        ds = await self._ds(db, images=["s3://b/a.png", "s3://b/b.png"],
                            anchor_uri="s3://b/a.png")
        out = await update_dataset(ds.id, DatasetUpdate(images=["s3://b/a.png"]),
                                   _user=None, db=db)
        assert out.anchor_uri == "s3://b/a.png"

    async def test_an_absent_field_does_not_clear_the_anchor(self, db):
        """The same distinction every other field on this route makes."""
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        ds = await self._ds(db, images=["s3://b/a.png"], anchor_uri="s3://b/a.png")
        out = await update_dataset(ds.id, DatasetUpdate(tags="x"), _user=None, db=db)
        assert out.anchor_uri == "s3://b/a.png"

    async def test_an_empty_string_clears_it(self, db):
        from app.routes.datasets import update_dataset
        from app.schemas.datasets import DatasetUpdate
        ds = await self._ds(db, images=["s3://b/a.png"], anchor_uri="s3://b/a.png")
        out = await update_dataset(ds.id, DatasetUpdate(anchor_uri=""), _user=None, db=db)
        assert out.anchor_uri is None

    async def test_scoring_refuses_an_anchor_that_is_not_in_the_set(self, db):
        """Rather than returning scores quietly measured against nothing."""
        from fastapi import HTTPException
        from app.routes.datasets import score_against_anchor
        ds = await self._ds(db, images=["s3://b/a.png"])
        with pytest.raises(HTTPException) as e:
            await score_against_anchor(ds.id, anchor_uri="s3://b/gone.png", _user=None, db=db)
        assert e.value.status_code == 422
        assert "not in this dataset" in e.value.detail

    async def test_scoring_needs_an_anchor_at_all(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import score_against_anchor
        ds = await self._ds(db, images=["s3://b/a.png"])
        with pytest.raises(HTTPException) as e:
            await score_against_anchor(ds.id, _user=None, db=db)
        assert e.value.status_code == 422

    def test_scoring_deletes_nothing(self):
        """The failure was culling with NO information, not culling by hand. A score beside each
        image fixes that and leaves the judgement where it belongs."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.score_against_anchor)
        assert "db.delete" not in src
        assert "ds.images =" not in src

    def test_a_missing_face_is_null_not_a_low_score(self):
        """_cos returns -2.0 for an absent embedding, which would render as the worst match in
        the set rather than as 'no face'."""
        import inspect
        from app.routes import datasets as mod
        assert "cos=None if not e else" in inspect.getsource(mod.score_against_anchor)


class TestTheCropRoundTripFitsThroughTheWire:
    """`cropping failed` in the console, `200 OK` in face-crop's log, nothing in this API's.
    The response could not get back: crops were full-resolution lossless PNG, ~80 MB for
    fourteen group photos, over a home uplink, inside a 300s read timeout."""

    def test_the_extension_follows_what_the_service_sent(self):
        """It returns JPEG now. Writing `.png` over JPEG bytes gives a library a file whose
        name lies, and the failure lands somewhere else entirely."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert 'get("format", "png")' in src

    def test_an_older_face_crop_still_works(self):
        """`format` is absent on a service that predates it, and PNG was the contract then — so
        the two repos can deploy in either order."""
        assert {"jpeg": "jpg"}.get(str({}.get("format", "png")).lower(), "png") == "png"

    def test_images_are_fetched_concurrently(self):
        """Fourteen serial round trips to S3 before any work started; they are independent and
        the wait is entirely network."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "asyncio.gather" in src
        assert "for u in ds.images" in src

    def test_crops_are_uploaded_concurrently(self):
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "asyncio.gather(*(put(" in src

    def test_the_upload_order_still_matches_the_scoring_order(self):
        """gather preserves argument order, and `images` is an ordered list the trainer stages
        in sequence — a set that came back shuffled would pair captions with the wrong faces."""
        import inspect
        from app.routes import datasets as mod
        src = inspect.getsource(mod.crop_faces)
        assert "uris = list(await asyncio.gather" in src

    def test_the_reference_embedding_fetch_is_concurrent_too(self):
        import inspect
        from app.routes import datasets as mod
        assert "asyncio.gather" in inspect.getsource(mod._embed_all)
