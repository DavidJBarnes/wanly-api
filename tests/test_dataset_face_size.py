"""Face size at training size, and "Fix small faces" (wanly-api#432).

Joana v3 had 29 of 48 faces under 250 px at TRAINING size -- the trainer scales a photo down to
the 1024^2 area and never enlarges a small one -- and v4, with head-and-shoulders crops and
upscaled close-ups added, learned her face in about half the steps (#431). These pin what the
API does with the face-crop service's /measure and upscale (wanly-gpu-docker#206):

  * which images the fix touches, and how (small image: upscaled in place; big photo with a
    small face: an upscaled crop ADDED beside it) -- and that pressing it twice adds nothing
  * that an older face-crop service is refused before anything is stored, never trusted
  * that the set is written once, at the end, against the row as it is then
"""
import base64
import json
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from app import face_size


def _imgs(n, prefix="fs"):
    return [f"s3://wanly-images/datasets/{prefix}/{i}.jpg" for i in range(n)]


# A measurement per image, encoded in the fake's bytes: the fake S3 hands back the URI, the
# fake service reads the size off SIZES. Upscaled and cropped outputs measure as the fix would
# leave them: ~1024 px with a big face.
SIZES: dict[str, tuple[int, int, float | None]] = {}


class _S3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []

    def download_bytes(self, uri):
        return self.objects.get(uri, uri.encode())

    def upload_bytes(self, data, key, bucket):
        uri = f"s3://{bucket}/{key}"
        self.objects[uri] = data
        self.uploads.append(uri)
        return uri


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


class _Service:
    """The face-crop service, as of #206 unless told it is older."""

    def __init__(self, old=False, ignores_upscale=False):
        self.old, self.ignores_upscale = old, ignores_upscale
        self.calls: list[str] = []
        self.crop_payloads: list[dict] = []

    def measure_one(self, b: bytes):
        if b.startswith(b"UP:") or b.startswith(b"CROP:"):
            return {"width": 819, "height": 1024, "train_scale": 1.0,
                    "faces": [{"face_h": 520.0, "face_px_at_train": 520.0, "yaw": 1.0,
                               "pitch": 2.0, "roll": 3.0, "det_score": 0.9}]}
        w, h, px = SIZES[b.decode()]
        faces = [] if px is None else [{"face_h": px, "face_px_at_train": px, "yaw": -12.0,
                                        "pitch": 4.0, "roll": 0.5, "det_score": 0.88}]
        return {"width": w, "height": h, "train_scale": 1.0, "faces": faces}

    def handle(self, method, path, body):
        self.calls.append(path)
        if path == "/health":
            feats = ["crop", "embed"] if self.old else ["crop", "embed", "measure", "upscale"]
            return _Resp({"status": "ok", **({} if self.old else {"features": feats})})
        imgs = [base64.b64decode(b) for b in (body or {}).get("images", [])]
        if path == "/measure":
            return _Resp({"detail": "Not Found"}, 404) if self.old else \
                _Resp({"results": [self.measure_one(b) for b in imgs]})
        if path == "/upscale":
            return _Resp({"detail": "Not Found"}, 404) if self.old else _Resp(
                {"images": [{"upscaled": True, "b64": base64.b64encode(b"UP:" + b).decode(),
                             "width": 697, "height": 1024} for b in imgs], "upscale": True})
        if path == "/embed":
            return _Resp({"embeddings": [[1.0, 0.0] for _ in imgs]})
        if path == "/crop":
            self.crop_payloads.append(body)
            out = {"faces": [{"source_index": i, "face_index": 0, "format": "jpeg",
                              "png_b64": base64.b64encode(b"CROP:" + b).decode(),
                              "embedding": [0.6, 0.8], "upscaled": True}
                             for i, b in enumerate(imgs)], "no_face": []}
            if not self.ignores_upscale:
                out.update(framing=body["framing"], upscale=body["upscale"])
            return _Resp(out)
        raise AssertionError(path)


def _wire(monkeypatch, svc: _Service, s3: _S3):
    from app.config import settings
    from app.routes import datasets as mod

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return svc.handle("GET", url.split("crop.test", 1)[1], None)

        async def post(self, url, json=None):
            return svc.handle("POST", url.split("crop.test", 1)[1], json)

    class _Httpx:
        AsyncClient = _Client
        HTTPError = httpx.HTTPError

    monkeypatch.setattr(face_size, "httpx", _Httpx)
    monkeypatch.setattr(face_size, "s3", s3)
    monkeypatch.setattr(mod, "s3", s3)
    monkeypatch.setattr(settings, "face_crop_url", "http://crop.test")


async def _ds(db, images, **kw):
    from app.models import Dataset
    kw.setdefault("captions", {})
    kw.setdefault("scores", {})
    d = Dataset(name=f"fs-{uuid.uuid4().hex[:6]}", images=list(images), prefix="datasets/fs",
                **kw)
    db.add(d)
    await db.commit()
    return d


class TestWhatCountsAsSmall:
    def test_under_the_line_at_training_size(self):
        assert face_size.is_small({"face_px": 249.9})
        assert not face_size.is_small({"face_px": 250.0})

    def test_no_face_and_unmeasured_are_not_small(self):
        """No face is the anchor scores' problem; unmeasured is unknown, not small."""
        assert not face_size.is_small({"face_px": None})
        assert not face_size.is_small(None)

    def test_a_small_image_is_by_its_short_side(self):
        assert face_size.is_small_image({"width": 254, "height": 373})
        assert not face_size.is_small_image({"width": 1080, "height": 1440})
        assert not face_size.is_small_image({"width": None, "height": None})

    def test_the_largest_face_is_the_subject(self):
        e = face_size.entry_from({"width": 1080, "height": 1440, "faces": [
            {"face_px_at_train": 300.0, "face_h": 350.0, "yaw": 5.0},
            {"face_px_at_train": 90.0}]})
        assert (e["face_px"], e["faces"], e["yaw"]) == (300.0, 2, 5.0)

    def test_no_face_is_stored_as_measured_with_none(self):
        e = face_size.entry_from({"width": 800, "height": 600, "faces": []})
        assert e["face_px"] is None and e["faces"] == 0 and e["width"] == 800
        assert face_size.entry_from(None)["face_px"] is None


class TestThePlan:
    IMGS = _imgs(5)

    def _faces(self):
        a, b, c, d, _ = self.IMGS
        return {a: {"width": 254, "height": 373, "face_px": 200.0},      # tiny close-up
                b: {"width": 1080, "height": 1440, "face_px": 120.0},    # small face, big photo
                c: {"width": 1080, "height": 1440, "face_px": 420.0},    # fine
                d: {"width": 400, "height": 500, "face_px": None}}       # small, no face
        # IMGS[4] is unmeasured

    def test_small_images_are_upscaled_and_big_photos_cropped(self):
        up, crop = face_size.plan_fix(self.IMGS, self._faces())
        assert up == [self.IMGS[0], self.IMGS[3]]
        assert crop == [self.IMGS[1]]

    def test_a_photo_whose_crop_is_already_in_the_set_is_not_cropped_again(self):
        """The photo stays small-faced forever; without this every press adds another copy."""
        faces = self._faces()
        faces[self.IMGS[1]]["crop_uri"] = "s3://wanly-images/datasets/fs/crop.jpg"
        _, crop = face_size.plan_fix(self.IMGS + ["s3://wanly-images/datasets/fs/crop.jpg"],
                                     faces)
        assert crop == []
        # ...unless that crop was removed since, in which case it is offered again.
        _, crop = face_size.plan_fix(self.IMGS, faces)
        assert crop == [self.IMGS[1]]

    def test_clips_are_never_touched(self):
        clip = "s3://wanly-images/datasets/fs/move.mp4"
        up, crop = face_size.plan_fix([clip], {clip: {"width": 100, "height": 100,
                                                      "face_px": 20.0}})
        assert up == [] and crop == []


@pytest.mark.asyncio
class TestMeasure:
    async def test_it_measures_the_unmeasured_stills(self, db, monkeypatch):
        from app.routes import datasets as mod
        imgs = _imgs(3, "m1")
        SIZES.update({imgs[0]: (1080, 1440, 180.0), imgs[1]: (300, 300, None)})
        ds = await _ds(db, imgs[:2] + ["s3://wanly-images/datasets/m1/c.mp4"],
                       faces={imgs[2]: {"face_px": 1.0}})
        svc, s3 = _Service(), _S3()
        _wire(monkeypatch, svc, s3)
        out = await mod.measure_faces(ds.id, overwrite=False, _user=None, db=db)
        assert out.faces[imgs[0]].face_px == 180.0 and out.faces[imgs[0]].yaw == -12.0
        assert out.faces[imgs[1]].face_px is None and out.faces[imgs[1]].width == 300
        # The clip is not measured; an entry for an image not in the set is not resurrected.
        assert set(out.faces) == {imgs[0], imgs[1]}

    async def test_a_remeasure_keeps_the_fix_bookkeeping(self, db, monkeypatch):
        from app.routes import datasets as mod
        imgs = _imgs(1, "m2")
        SIZES[imgs[0]] = (1080, 1440, 120.0)
        ds = await _ds(db, imgs, faces={imgs[0]: {"face_px": 1.0, "crop_uri": "s3://x/c.jpg"}})
        _wire(monkeypatch, _Service(), _S3())
        out = await mod.measure_faces(ds.id, overwrite=True, _user=None, db=db)
        assert out.faces[imgs[0]].face_px == 120.0
        assert out.faces[imgs[0]].crop_uri == "s3://x/c.jpg"

    async def test_an_older_service_is_a_503_that_says_what_it_needs(self, db, monkeypatch):
        from fastapi import HTTPException
        from app.routes import datasets as mod
        imgs = _imgs(1, "m3")
        SIZES[imgs[0]] = (1080, 1440, 120.0)
        ds = await _ds(db, imgs)
        _wire(monkeypatch, _Service(old=True), _S3())
        with pytest.raises(HTTPException) as e:
            await mod.measure_faces(ds.id, overwrite=False, _user=None, db=db)
        assert e.value.status_code == 503 and "wanly-gpu-docker#206" in e.value.detail
        await db.refresh(ds)
        assert ds.faces == {}

    async def test_the_response_carries_faces(self, db):
        """Routes hand-build the response; a field missing from the schema drops silently."""
        from app.routes.datasets import _respond
        imgs = _imgs(1, "m4")
        ds = await _ds(db, imgs, faces={imgs[0]: {"face_px": 99.0, "width": 10, "height": 10}})
        body = json.loads(_respond(ds, None).model_dump_json())
        assert body["faces"][imgs[0]]["face_px"] == 99.0

    async def test_removing_an_image_drops_its_measurement(self, db):
        from app.routes.datasets import _prune_annotations
        imgs = _imgs(2, "m5")
        ds = await _ds(db, imgs, faces={u: {"face_px": 300.0} for u in imgs})
        ds.images = imgs[:1]
        _prune_annotations(ds)
        assert set(ds.faces) == {imgs[0]}

    async def test_a_clone_keeps_the_measurements(self):
        import inspect
        from app.routes import datasets as mod
        assert "faces=dict(src.faces" in inspect.getsource(mod.clone_dataset)


@pytest.fixture
def shared_session(db, monkeypatch):
    from app.routes import datasets as mod

    @asynccontextmanager
    async def _session():
        yield db
    monkeypatch.setattr(mod, "async_session", _session)
    return db


@pytest.mark.asyncio
class TestFixSmallFaces:
    async def _world(self, db, monkeypatch, svc=None, prefix="f"):
        imgs = _imgs(4, prefix)
        tiny, far, fine, noface = imgs
        SIZES.update({tiny: (254, 373, 200.0), far: (1080, 1440, 120.0),
                      fine: (1080, 1440, 420.0), noface: (1080, 1440, None)})
        ds = await _ds(db, imgs, anchor_uri=tiny,
                       captions={tiny: "close-up", far: "standing outside"},
                       scores={tiny: 1.0, far: 0.7, fine: 0.8, noface: None})
        svc = svc or _Service()
        s3 = _S3()
        _wire(monkeypatch, svc, s3)
        face_size.FIX_RUNS.pop(ds.id, None)
        return ds, imgs, svc, s3

    async def test_it_upscales_in_place_and_adds_crops(self, db, monkeypatch, shared_session):
        from app.routes import datasets as mod
        ds, (tiny, far, fine, noface), svc, s3 = await self._world(db, monkeypatch)
        await mod.fix_small_faces_job(ds.id)
        run = face_size.FIX_RUNS[ds.id]
        assert run["error"] is None and run["running"] is False and run["stage"] == "done"
        await db.refresh(ds)
        up = [u for u in s3.uploads if "/upscaled-" in u]
        crops = [u for u in s3.uploads if "/portraits-" in u]
        assert len(up) == 1 and len(crops) == 1
        # In place, same position; the crop joins the end; the original photo stays.
        assert ds.images == [up[0], far, fine, noface, crops[0]]
        # The upscaled copy IS the photograph: its caption, score and anchor role move to it.
        assert ds.anchor_uri == up[0]
        assert ds.captions == {up[0]: "close-up", far: "standing outside"}
        assert ds.scores[up[0]] == 1.0
        # The crop arrives scored against the anchor, from its own embedding.
        assert ds.scores[crops[0]] == pytest.approx(0.6)
        # Measured, with the bookkeeping that makes a second press a no-op.
        assert ds.faces[up[0]]["face_px"] == 520.0
        assert ds.faces[up[0]]["upscaled_from"] == tiny
        assert ds.faces[far]["crop_uri"] == crops[0]
        assert tiny not in ds.faces
        # The crop request asked for exactly the fix.
        p = svc.crop_payloads[0]
        assert (p["framing"], p["upscale"], p["largest_only"]) == ("head_shoulders", True, True)
        assert "Fixed small faces" in ds.notes and run["summary"] in ds.notes

    async def test_pressing_it_twice_adds_nothing(self, db, monkeypatch, shared_session):
        from app.routes import datasets as mod
        ds, *_ = await self._world(db, monkeypatch, prefix="f2")
        await mod.fix_small_faces_job(ds.id)
        await db.refresh(ds)
        before = list(ds.images)
        await mod.fix_small_faces_job(ds.id)
        await db.refresh(ds)
        assert ds.images == before
        assert face_size.FIX_RUNS[ds.id]["summary"] == "No small faces to fix."

    async def test_a_service_that_ignores_upscale_stores_nothing(self, db, monkeypatch,
                                                                 shared_session):
        """It sends plain crops. Stored, they would join the set as if they were the fix."""
        from app.routes import datasets as mod
        ds, imgs, *_ = await self._world(db, monkeypatch, _Service(ignores_upscale=True), "f3")
        await mod.fix_small_faces_job(ds.id)
        assert "wanly-gpu-docker#206" in face_size.FIX_RUNS[ds.id]["error"]
        await db.refresh(ds)
        assert ds.images == imgs and ds.anchor_uri == imgs[0]

    async def test_an_image_removed_while_it_ran_stays_removed(self, db, monkeypatch,
                                                               shared_session):
        from app.routes import datasets as mod
        ds, (tiny, far, fine, noface), svc, s3 = await self._world(db, monkeypatch, prefix="f4")
        real = face_size.run_fix

        async def remove_far_meanwhile(*a, **kw):
            out = await real(*a, **kw)
            ds.images = [tiny, fine, noface]
            await db.commit()
            return out
        monkeypatch.setattr(face_size, "run_fix", remove_far_meanwhile)
        await mod.fix_small_faces_job(ds.id)
        await db.refresh(ds)
        assert far not in ds.images
        assert not [u for u in ds.images if "/portraits-" in u]

    async def test_an_older_service_is_refused_before_anything_starts(self, db, monkeypatch):
        from fastapi import BackgroundTasks, HTTPException
        from app.routes import datasets as mod
        ds, *_ = await self._world(db, monkeypatch, _Service(old=True), "f5")
        tasks = BackgroundTasks()
        with pytest.raises(HTTPException) as e:
            await mod.fix_small_faces(ds.id, tasks, _user=None, db=db)
        assert e.value.status_code == 503 and "wanly-gpu-docker#206" in e.value.detail
        assert not tasks.tasks and ds.id not in face_size.FIX_RUNS

    async def test_a_locked_set_is_refused(self, db, monkeypatch):
        from datetime import datetime, timezone
        from fastapi import BackgroundTasks, HTTPException
        from app.routes import datasets as mod
        ds, *_ = await self._world(db, monkeypatch, prefix="f6")
        ds.locked_at = datetime.now(timezone.utc)
        await db.commit()
        with pytest.raises(HTTPException) as e:
            await mod.fix_small_faces(ds.id, BackgroundTasks(), _user=None, db=db)
        assert e.value.status_code == 409

    async def test_it_starts_in_the_background_and_reports_progress(self, db, monkeypatch):
        from fastapi import BackgroundTasks
        from app.routes import datasets as mod
        ds, *_ = await self._world(db, monkeypatch, prefix="f7")
        tasks = BackgroundTasks()
        out = await mod.fix_small_faces(ds.id, tasks, _user=None, db=db)
        assert out.running is True and len(tasks.tasks) == 1
        # A second press while it runs returns that run, and starts nothing.
        again = BackgroundTasks()
        assert (await mod.fix_small_faces(ds.id, again, _user=None, db=db)).running
        assert not again.tasks
        face_size.FIX_RUNS.pop(ds.id, None)


class _CountingS3(_S3):
    """Counts downloads, so a test can see how many images were in hand at each service call."""

    def __init__(self, missing=()):
        super().__init__()
        self.downloads = 0
        self.missing = set(missing)

    def download_bytes(self, uri):
        if uri in self.missing:
            raise FileNotFoundError(uri)
        self.downloads += 1
        return super().download_bytes(uri)


class _CountingService(_Service):
    """Records, at every image-carrying call, how many images had been downloaded since the
    previous one -- i.e. how many were held to make this call."""

    def __init__(self, s3: _CountingS3):
        super().__init__()
        self.s3, self.seen, self.held = s3, 0, []

    def handle(self, method, path, body):
        if body and body.get("images"):
            self.held.append(self.s3.downloads - self.seen)
            self.seen = self.s3.downloads
        return super().handle(method, path, body)


@pytest.mark.asyncio
class TestOneChunkInHand:
    """#437/#434: never a whole set's bytes in hand. test_face_size_bounded.py pins measure's
    lock and download-failure handling; this pins the per-call shape, for the fix too."""

    async def test_measure_downloads_one_chunk_per_call(self, monkeypatch):
        imgs = _imgs(40, "c1")
        SIZES.update({u: (1080, 1440, 300.0) for u in imgs})
        s3 = _CountingS3()
        svc = _CountingService(s3)
        _wire(monkeypatch, svc, s3)
        out = await face_size.measure_uris(imgs)
        assert set(out) == set(imgs)
        assert svc.held == [16, 16, 8]

    async def test_a_fix_holds_one_chunk_at_a_time(self, monkeypatch):
        tiny, far = _imgs(6, "c5"), _imgs(9, "c6")
        SIZES.update({u: (254, 373, 200.0) for u in tiny})
        SIZES.update({u: (1080, 1440, 120.0) for u in far})
        faces = {u: {"width": 254, "height": 373, "face_px": 200.0} for u in tiny}
        faces.update({u: {"width": 1080, "height": 1440, "face_px": 120.0} for u in far})
        s3 = _CountingS3()
        svc = _CountingService(s3)
        _wire(monkeypatch, svc, s3)
        ds_id = uuid.uuid4()
        out = await face_size.run_fix(ds_id, tiny + far, faces, "datasets/c", None)
        face_size.FIX_RUNS.pop(ds_id, None)
        assert len(out["replaced"]) == 6 and len(out["added"]) == 9
        # upscale 4+2, crop 4+4+1, then the results measured in one chunk of 15.
        assert svc.held == [4, 2, 4, 4, 1, 15]


@pytest.mark.asyncio
class TestOneFixAtATime:
    async def test_a_second_fix_waits_and_says_so(self, db, monkeypatch, shared_session):
        import asyncio
        from app.routes import datasets as mod
        imgs = _imgs(2, "w1")
        SIZES.update({imgs[0]: (254, 373, 200.0), imgs[1]: (1080, 1440, 420.0)})
        ds = await _ds(db, imgs)
        _wire(monkeypatch, _Service(), _S3())
        face_size.FIX_RUNS.pop(ds.id, None)
        async with face_size.FIX_LOCK:
            task = asyncio.create_task(mod.fix_small_faces_job(ds.id))
            await asyncio.sleep(0.01)
            assert face_size.FIX_RUNS[ds.id]["stage"] == "waiting for another fix"
            await db.refresh(ds)
            assert ds.images == imgs
        await task
        assert face_size.FIX_RUNS[ds.id]["stage"] == "done"
        await db.refresh(ds)
        assert any("/upscaled-" in u for u in ds.images)
