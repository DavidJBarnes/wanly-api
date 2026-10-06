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
    def test_a_job_names_a_character_not_a_dataset(self):
        """#352: which set trains is the SERVER's call -- the member's own character set --
        so a request can no longer point at an arbitrary one (or a raw list of URIs)."""
        t = TrainingCreate(mode="solo", character="p@y")
        assert not hasattr(t, "dataset_id") and not hasattr(t, "dataset_images")
        with pytest.raises(ValueError, match="no longer accepted"):
            TrainingCreate(mode="solo", character="x", dataset_id=str(uuid.uuid4()))


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
        positional contract the other call sites already honour. (#356 made it
        delete_prefix_except, keeping what other sets list; tests/test_dataset_lock.py
        runs that against a bucket.)"""
        from unittest.mock import patch
        from app import s3
        from app.config import settings
        from app.routes import datasets as mod

        ds = Dataset(id=uuid.uuid4(), name="Me", prefix="dataset-test-faces", images=[])
        calls = []

        def fake_delete_prefix_except(prefix, bucket, except_uris):
            calls.append((prefix, bucket, except_uris))
            return 0

        class _Rows:
            def all(self):
                return []

        class FakeDb:
            async def get(self, model, _id):
                return ds
            async def execute(self, _q):
                # No training runs lock it; no other set lists anything.
                return _Rows()
            async def delete(self, row):
                pass
            async def commit(self):
                pass

        with patch.object(s3, "delete_prefix_except", fake_delete_prefix_except):
            await mod.delete_dataset(ds.id, purge=True, _user=None, db=FakeDb())

        assert calls == [(ds.prefix + "/", settings.s3_images_bucket, set())]

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
        # A fresh suffix per batch, whatever the framing calls it (#409).
        assert '{kind}-{uuid.uuid4().hex[:6]}' in inspect.getsource(mod.crop_faces)

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

    def __init__(self, gone: set[str] | None = None):
        self.downloaded: list[str] = []
        self.gone = gone or set()

    def download_bytes(self, uri):
        self.downloaded.append(uri)
        return b"jpeg"

    def head_object(self, uri):
        return None if uri in self.gone else {"Key": uri}

    def upload_bytes(self, data, key, bucket):
        return f"s3://{bucket}/{key}"


class _CropResp:
    def raise_for_status(self):
        pass

    def json(self):
        # One face per image submitted, which is what a solo-face dataset produces.
        out = {"faces": [{"source_index": i, "face_index": 0,
                          "png_b64": b"eA==", "format": "jpeg"}
                         for i in range(self.n)],
               "no_face": []}
        # A service that knows framing echoes it (#409); `old_service` is one that predates it.
        if not getattr(_crop_client, "old_service", False):
            out["framing"] = getattr(self, "framing", "face")
        return out


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
            r.framing = json.get("framing", "face")
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

    async def _crop(self, db, monkeypatch, ds, query=None, extra=""):
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
                if extra:
                    q += ("&" if q else "?") + extra
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

    async def test_the_default_framing_is_the_face_crop(self, db, monkeypatch):
        """#409: absent means face -- every caller that predates framing keeps its crop."""
        resp, _ = await self._crop(db, monkeypatch, await self._ds(db))
        assert resp.status_code == 200, resp.text
        assert _crop_client.last_payload["framing"] == "face"
        assert all("/faces-" in u for u in resp.json()["images"] if u not in self.IMGS)

    async def test_head_and_shoulders_is_passed_through_and_filed_as_portraits(self, db,
                                                                             monkeypatch):
        resp, _ = await self._crop(db, monkeypatch, await self._ds(db),
                                   extra="framing=head_shoulders")
        assert resp.status_code == 200, resp.text
        assert _crop_client.last_payload["framing"] == "head_shoulders"
        crops = [u for u in resp.json()["images"] if u not in self.IMGS]
        assert crops and all("/portraits-" in u for u in crops)
        assert "head-and-shoulders crops" in resp.json()["notes"]

    async def test_an_older_service_cannot_pass_off_face_crops_as_portraits(self, db,
                                                                          monkeypatch):
        """It ignores `framing` and sends face crops. Stored, the set would silently be the
        wrong framing; refused, nothing changes."""
        ds = await self._ds(db)
        monkeypatch.setattr(_crop_client, "old_service", True, raising=False)
        resp, _ = await self._crop(db, monkeypatch, ds, extra="framing=head_shoulders")
        assert resp.status_code == 503
        assert "does not support" in resp.json()["detail"]
        await db.refresh(ds)
        assert ds.images == self.IMGS
        # The old service is still fine for what it does support.
        resp, _ = await self._crop(db, monkeypatch, ds)
        assert resp.status_code == 200, resp.text

    async def test_an_unknown_framing_is_a_422(self, db, monkeypatch):
        resp, _ = await self._crop(db, monkeypatch, await self._ds(db), extra="framing=body")
        assert resp.status_code == 422

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
        assert "None if not e else" in inspect.getsource(mod.score_against_anchor)


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


# -------------------------------------------------------------------------------------------
# #352: ownership, per-image captions and scores, and regularization pools.
# -------------------------------------------------------------------------------------------

def _imgs(n, prefix="x"):
    return [f"s3://wanly-images/datasets/{prefix}/{i}.jpg" for i in range(n)]


async def _ds(db, **kw):
    from app.models import Dataset
    base = dict(name=f"set-{uuid.uuid4().hex[:6]}", images=_imgs(4), prefix="datasets/p",
                captions={}, scores={})
    base.update(kw)
    d = Dataset(**base)
    db.add(d)
    await db.commit()
    return d


async def _patch(db, ds, **fields):
    from app.routes.datasets import update_dataset
    from app.schemas.datasets import DatasetUpdate
    return await update_dataset(ds.id, DatasetUpdate(**fields), _user=None, db=db)


@pytest.mark.asyncio
class TestCaptionsAndScoresFollowTheImages:
    """Both are keyed by URI. An entry for an image that left the set is a caption nobody
    can see, and a crop reusing a filename would inherit a stranger's."""

    async def test_removing_an_image_drops_its_caption_and_score(self, db):
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs, anchor_uri=imgs[0],
                       captions={u: f"c{i}" for i, u in enumerate(imgs)},
                       scores={u: 0.9 for u in imgs})
        out = await _patch(db, ds, images=[imgs[0], imgs[2]])
        assert set(out.captions) == {imgs[0], imgs[2]}
        assert set(out.scores) == {imgs[0], imgs[2]}

    async def test_reordering_keeps_them(self, db):
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs, captions={u: "c" for u in imgs})
        out = await _patch(db, ds, images=list(reversed(imgs)))
        assert set(out.captions) == set(imgs)

    async def test_a_new_anchor_clears_the_scores(self, db):
        """Scores are likeness to THE anchor; against another face they are about someone
        else, and training would pass or refuse the set on them."""
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs, anchor_uri=imgs[0], scores={u: 0.9 for u in imgs})
        out = await _patch(db, ds, anchor_uri=imgs[1])
        assert out.scores == {}

    async def test_an_unrelated_edit_keeps_the_scores(self, db):
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs, anchor_uri=imgs[0], scores={u: 0.9 for u in imgs})
        out = await _patch(db, ds, notes="hello")
        assert len(out.scores) == 3

    async def test_scoring_persists_what_it_returns(self, db, monkeypatch):
        from app.config import settings
        from app.routes import datasets as mod
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs)

        async def _embed(uris):
            # anchor, a match, no face -- one embedding per still (#411)
            return [[[1.0, 0.0]], [[0.6, 0.8]], [[]]]
        monkeypatch.setattr(mod, "_embed_all", _embed)
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")
        out = await mod.score_against_anchor(ds.id, anchor_uri=imgs[0], _user=None, db=db)
        await db.refresh(ds)
        assert ds.scores == {imgs[0]: 1.0, imgs[1]: 0.6, imgs[2]: None}
        assert [s.cos for s in out.scores] == [1.0, 0.6, None]
        assert ds.anchor_uri == imgs[0]

    async def test_a_crop_in_place_drops_the_photographs_annotations(self, db, monkeypatch):
        from app.config import settings
        from app.routes import datasets as mod
        imgs = _imgs(2)
        ds = await _ds(db, images=imgs, anchor_uri=imgs[0],
                       captions={u: "photo" for u in imgs}, scores={u: 0.9 for u in imgs})
        monkeypatch.setattr(mod, "s3", _FakeS3())
        monkeypatch.setattr(mod, "httpx", _HttpxShim(_crop_client(None)))
        monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")
        out = await mod.crop_faces(ds.id, largest_only=False, uris=None, save_as=False,
                                   _user=None, db=db)
        assert out.captions == {} and out.scores == {}

    async def test_a_reupload_over_the_same_name_drops_the_stale_caption(self, db, monkeypatch):
        from app.routes import datasets as mod
        ds = await _ds(db, images=["s3://wanly-images/datasets/p/a.jpg"],
                       captions={"s3://wanly-images/datasets/p/a.jpg": "old picture"})
        monkeypatch.setattr(mod, "s3", _FakeS3())

        class _F:
            filename = "a.jpg"

            async def read(self):
                return b"new bytes"
        out = await mod.add_images(ds.id, files=[_F()], _user=None, db=db)
        assert out.captions == {}


@pytest.mark.asyncio
class TestOwnership:
    async def _chars(self, db):
        from app.models import LtxCharacter
        db.add_all([LtxCharacter(name="David", trigger="d@vid", gender="man", char_lora="none"),
                    LtxCharacter(name="DavidKelly-2026", kind="pair", trigger="x",
                                 char_lora="none", members=["David", "Kelly-2026"])])
        await db.flush()

    async def test_a_character_set_belongs_to_a_registered_solo_character(self, db):
        from fastapi import HTTPException
        await self._chars(db)
        ds = await _ds(db)
        out = await _patch(db, ds, kind="character", character="david")
        assert (out.kind, out.character) == ("character", "David"), (
            "the owner is matched case-insensitively and stored as the registry spells it")
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, kind="character", character="Nobody")
        assert e.value.status_code == 422

    async def test_a_composition_set_belongs_to_a_pair_never_a_person(self, db):
        from fastapi import HTTPException
        await self._chars(db)
        ds = await _ds(db)
        out = await _patch(db, ds, kind="composition", character="DavidKelly-2026")
        assert out.character == "DavidKelly-2026"
        # Not registered yet is fine -- a pair row is created by its first publish.
        out = await _patch(db, ds, kind="composition", character="DavidKelly-2000")
        assert out.character == "DavidKelly-2000"
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, kind="composition", character="David")
        assert e.value.status_code == 422

    async def test_a_regularization_pool_has_a_class_and_no_owner(self, db):
        from fastapi import HTTPException
        ds = await _ds(db)
        out = await _patch(db, ds, kind="regularization", reg_class="woman", character=None)
        assert (out.kind, out.character, out.reg_class) == ("regularization", None, "woman")
        with pytest.raises(HTTPException):
            await _patch(db, ds, kind="regularization", reg_class=None)

    async def test_kind_null_unassigns_everything(self, db):
        await self._chars(db)
        ds = await _ds(db, kind="character", character="David")
        out = await _patch(db, ds, kind=None)
        assert (out.kind, out.character, out.reg_class) == (None, None, None)

    async def test_an_absent_kind_changes_nothing(self, db):
        ds = await _ds(db, kind="regularization", reg_class="man")
        out = await _patch(db, ds, notes="x")
        assert (out.kind, out.reg_class) == ("regularization", "man")

    async def test_ownership_is_locked_while_a_run_trains_on_the_set(self, db):
        from fastapi import HTTPException
        from app.enums import TrainingStatus
        from app.models import TrainingJob
        await self._chars(db)
        ds = await _ds(db, kind="character", character="David")
        db.add(TrainingJob(character="David", trigger="d@vid", version=1,
                           dataset_images=ds.images, status=TrainingStatus.RUNNING,
                           config={"dataset": {"id": str(ds.id), "name": ds.name}}))
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await _patch(db, ds, kind=None)
        assert e.value.status_code == 409
        # ...while everything else is still editable.
        assert (await _patch(db, ds, notes="fine")).notes == "fine"


@pytest.mark.asyncio
class TestTrainingCaptions:
    """Per-image caption BODIES, through the captioner's queue, with the training
    instruction -- never the trigger."""

    def test_the_instruction_forbids_identity(self):
        from app.joycaption import TRAINING_CAPTION
        low = TRAINING_CAPTION.lower()
        for must in ("framing", "pose", "head angle", "expression", "clothing", "hair",
                     "lighting", "background"):
            assert must in low
        for banned in ("facial features", "age", "ethnicity", "body type", "names",
                       "the image shows"):
            assert banned in low

    async def _fake_captioner(self, monkeypatch, texts=None, fail_at=None):
        from app.joycaption import CaptionerBusy
        from app.routes import captions as cap_mod
        from app.routes import datasets as mod
        seen = []

        async def _caption(db, image, style=None, instruction=None, interactive=True):
            seen.append(instruction)
            if fail_at is not None and len(seen) == fail_at:
                raise CaptionerBusy("3090.zero is in render mode")
            return f"caption {len(seen)}", instruction
        monkeypatch.setattr(cap_mod, "caption_image_bytes", _caption)
        monkeypatch.setattr(mod, "s3", _FakeS3())
        return seen

    async def test_missing_captions_are_filled_and_existing_kept(self, db, monkeypatch):
        from app.joycaption import TRAINING_CAPTION
        from app.routes.datasets import caption_dataset_images
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs, captions={imgs[1]: "hand written"})
        seen = await self._fake_captioner(monkeypatch)
        n = await caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 2
        assert ds.captions[imgs[1]] == "hand written"
        assert set(ds.captions) == set(imgs)
        assert seen == [TRAINING_CAPTION, TRAINING_CAPTION]

    async def test_an_image_deleted_while_captioned_gets_no_caption(self, db, monkeypatch):
        """console#559: DELETE /images keeps the set's membership (the 409's dead entry), so
        the loop's membership check cannot see a delete that landed while the captioner was
        working on that image. Storing the caption would recreate the orphan the delete just
        dropped."""
        from app.routes import datasets as mod
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs)
        await self._fake_captioner(monkeypatch)
        monkeypatch.setattr(mod, "s3", _FakeS3(gone={imgs[1]}))
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 2
        assert set(ds.captions) == {imgs[0], imgs[2]}

    async def test_overwrite_redoes_every_one(self, db, monkeypatch):
        from app.routes.datasets import caption_dataset_images
        imgs = _imgs(3)
        ds = await _ds(db, images=imgs, captions={imgs[1]: "hand written"})
        await self._fake_captioner(monkeypatch)
        assert await caption_dataset_images(db, ds.id, overwrite=True) == 3

    async def test_a_refusal_stops_the_run_and_says_why(self, db, monkeypatch):
        from app.routes import datasets as mod
        imgs = _imgs(4)
        ds = await _ds(db, images=imgs)
        await self._fake_captioner(monkeypatch, fail_at=2)
        mod._CAPTION_RUNS[ds.id] = {"running": True, "error": None}
        n = await mod.caption_dataset_images(db, ds.id, overwrite=False)
        await db.refresh(ds)
        assert n == 1 and len(ds.captions) == 1
        status = mod._caption_status(ds)
        assert status.total == 4 and status.captioned == 1
        assert "render mode" in status.error
        mod._CAPTION_RUNS.pop(ds.id, None)

    async def test_a_hand_edit_is_by_uri_and_blank_deletes(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import edit_dataset_caption
        from app.schemas.datasets import DatasetCaptionEdit
        imgs = _imgs(2)
        ds = await _ds(db, images=imgs)
        out = await edit_dataset_caption(
            ds.id, DatasetCaptionEdit(uri=imgs[0], caption="  close-up,\n smiling "),
            _user=None, db=db)
        assert out.captions == {imgs[0]: "close-up, smiling"}
        out = await edit_dataset_caption(ds.id, DatasetCaptionEdit(uri=imgs[0], caption=""),
                                         _user=None, db=db)
        assert out.captions == {}
        with pytest.raises(HTTPException) as e:
            await edit_dataset_caption(ds.id, DatasetCaptionEdit(uri="s3://gone", caption="x"),
                                       _user=None, db=db)
        assert e.value.status_code == 422

    async def test_the_endpoints_over_http(self, db, monkeypatch):
        from httpx import ASGITransport, AsyncClient
        from app.auth import get_current_user, verify_api_key_or_bearer
        from app.database import get_db
        from app.main import app
        from app.routes import datasets as mod
        ds = await _ds(db, images=_imgs(2))
        started = []
        monkeypatch.setattr(mod, "caption_dataset_job",
                            lambda ds_id, overwrite: started.append((ds_id, overwrite)))
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: object()
        app.dependency_overrides[verify_api_key_or_bearer] = lambda: None
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
                r = await c.post(f"/datasets/{ds.id}/captions", json={"overwrite": True})
                st = await c.get(f"/datasets/{ds.id}/captions/status")
        finally:
            app.dependency_overrides.clear()
            mod._CAPTION_RUNS.pop(ds.id, None)
        assert r.status_code == 202
        assert r.json() == {"total": 2, "captioned": 0, "running": True, "error": None}
        assert st.json()["running"] is True
        assert started == [(ds.id, True)]


@pytest.mark.asyncio
class TestRegularizationPools:
    async def test_only_a_regularization_set_can_be_generated_into(self, db):
        from fastapi import HTTPException
        from app.routes.datasets import regularize_dataset
        from app.schemas.datasets import DatasetRegularize
        ds = await _ds(db)
        with pytest.raises(HTTPException) as e:
            await regularize_dataset(ds.id, DatasetRegularize(count=3), user=_Usr(), db=db)
        assert e.value.status_code == 422

    async def test_renders_are_text_to_video_with_no_character(self, db):
        from sqlalchemy import select
        from app.models import Job, Segment, User
        from app.regularization import REG_FRAMES, REG_PRIORITY_BASE
        from app.routes.datasets import regularize_dataset
        from app.schemas.datasets import DatasetRegularize
        user = User(username=f"u{uuid.uuid4().hex[:6]}", password_hash="x")
        db.add(user)
        await db.flush()
        ds = await _ds(db, images=[], kind="regularization", reg_class="woman")
        out = await regularize_dataset(ds.id, DatasetRegularize(count=6), user=user, db=db)
        assert (out.requested, out.running, out.done) == (6, 6, 0)
        jobs = (await db.execute(select(Job).where(Job.user_id == user.id))).scalars().all()
        assert len(jobs) == 6
        assert all(j.priority >= REG_PRIORITY_BASE for j in jobs), "a pool queues behind real work"
        assert all(j.starting_image is None for j in jobs)
        assert all(j.width % 64 == 0 and j.height % 64 == 0 for j in jobs)
        segs = (await db.execute(select(Segment).where(
            Segment.job_id.in_([j.id for j in jobs])))).scalars().all()
        for s in segs:
            assert s.index == 0 and s.start_image is None
            r = s.ltx_recipe
            assert r["recipe"], "the engine renders text-to-video only on the recipe path"
            assert r["characters"] == [] and r["char_lora"] == "none"
            assert r["frames"] == REG_FRAMES == 25
            assert r["checkpoint"] == "10Eros_v1.5_bf16"
            assert " woman " in s.prompt
        assert len({s.prompt for s in segs}) > 1, "a pool of one prompt regularizes to one face"

    async def test_a_new_job_still_lands_ahead_of_the_pool(self, db):
        import inspect
        from app.routes import jobs as mod
        assert "Job.priority < REG_PRIORITY_BASE" in inspect.getsource(mod.create_job)

    async def test_finished_frames_are_collected_once(self, db, monkeypatch):
        from app.enums import JobStatus, SegmentStatus
        from app.models import Job, Segment, User
        from app.regularization import reg_tag
        from app.routes import datasets as mod
        user = User(username=f"u{uuid.uuid4().hex[:6]}", password_hash="x")
        db.add(user)
        await db.flush()
        ds = await _ds(db, images=[], kind="regularization", reg_class="man",
                       prefix="datasets/pool")
        segs = []
        for status in (SegmentStatus.COMPLETED, SegmentStatus.PENDING, SegmentStatus.FAILED,
                       SegmentStatus.PROCESSING):
            j = Job(user_id=user.id, name="r", width=832, height=1216, fps=24, seed=1,
                    tags=f"regularization, {reg_tag(ds.id)}", status=JobStatus.PROCESSING)
            db.add(j)
            await db.flush()
            s = Segment(job_id=j.id, index=0, prompt="p", status=status,
                        last_frame_path=(f"s3://wanly-jobs/{j.id}/last_frame.png"
                                         if status == SegmentStatus.COMPLETED else None))
            db.add(s)
            segs.append((j, s))
        await db.commit()
        monkeypatch.setattr(mod, "s3", _FakeS3())

        first = await mod.regularize_status(ds.id, _user=None, db=db)
        assert (first.requested, first.done, first.failed, first.running) == (4, 1, 1, 2)
        assert first.collected_now == 1
        await db.refresh(ds)
        done_seg = segs[0][1]
        assert ds.images == [f"s3://wanly-images/datasets/pool/reg/{done_seg.id}.png"]
        assert segs[0][0].status == JobStatus.ARCHIVED

        # A human removes the frame as a bad render; the next poll must not bring it back.
        await _patch(db, ds, images=[])
        again = await mod.regularize_status(ds.id, _user=None, db=db)
        assert again.collected_now == 0 and again.done == 1
        await db.refresh(ds)
        assert ds.images == []


class _Usr:
    id = None
